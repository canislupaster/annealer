#include <atomic>
#include <barrier>
#include <chrono>
#include <iostream>
#include <fstream>
#include <concepts>
#include <iterator>
#include <limits>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <random>
#include <thread>
#include <algorithm>
#include <format>
#include <functional>
#include <ranges>
#include <map>

#include "annealer_old.hpp"
#include "parser.hh"

using namespace std;

void make_triangular(vector<vector<double>>& mat) {
	for (int i=0; i<mat.size(); i++) {
		for (int j=i+1; j<mat[i].size(); j++) {
			mat[j][i]+=mat[i][j];
			mat[i][j]=0;
		}
	}
}

struct QUBO {
	int n=0;

	QUBO(vector<pair<array<int,2>, double>>&& sparse): entries(sparse) {
		for (auto [x,y]: entries) {
			if (x[1]>x[0]) throw runtime_error("matrix is not lower triangular");
			n = max({n,x[0]+1,x[1]+1});
		}

		adj.resize(n), diag.resize(n);

		for (auto [x,y]: entries) {
			if (x[0]!=x[1]) {
				adj[x[0]].emplace_back(x[1],y);
				adj[x[1]].emplace_back(x[0],y);
			} else {
				diag[x[0]]=y;
			}
		}
	}

	static vector<pair<array<int,2>, double>> dense_to_sparse(vector<vector<double>> const& dense) {
		vector<pair<array<int,2>,double>> sparse;

		if (dense.size() != dense[0].size())
			throw runtime_error("matrix is not square");

		for (int i = 0; i < dense.size(); i++) {
			for (int j = 0; j<dense.size(); j++) {
				if (abs(dense[i][j])>1e-9)
					sparse.emplace_back(array<int,2>{i,j}, dense[i][j]);
			}
		}

		return sparse;
	}

	QUBO(vector<vector<double>> dense): QUBO(dense_to_sparse(dense)) {}

	vector<double> diag;
	vector<vector<pair<int,double>>> adj;
	vector<pair<array<int,2>, double>> entries;

	friend ostream& operator<<(ostream& os, QUBO const& q) {
		for (const auto& entry : q.entries) {
			os << '[' << entry.first[0] << ' ' << entry.first[1] << "] : " << entry.second << endl;
		}
		return os;
	}

	double diff(int flip, vector<bool> const& solution) const {
		double out=diag[flip];
		for (auto [a,b]: adj[flip])
			if (solution[a]) out+=b;
		
		return solution[flip] ? -out : out;
	}
};

struct Settings {
	int max_iter, nthread, synchronize_interval;
	int stop_threshold, restart_threshold;
	float T_0, alpha;
	unsigned seed;

	//print settings
	friend ostream& operator<<(ostream& os, Settings const& s) {
		return os<<format(
			"max_iter={}, nthread={}, synchronize_interval={}, stop_threshold={}, restart_threshold={}, T_0={}, alpha={}, seed={}",
			s.max_iter, s.nthread, s.synchronize_interval, s.stop_threshold, s.restart_threshold, s.T_0, s.alpha, s.seed
		);
	}
};

ostream& operator<<(ostream& os, const vector<bool>& x) {
	for (auto xi : x) os << xi << ' ';
	return os;
}

struct Pool {
	vector<thread> threads;
	barrier<> done;
	function<void(int)> f;
	atomic<int> left;
	bool ex=false;

	Pool(int nthread=thread::hardware_concurrency()-1): done(nthread+1) {
		while (nthread--) threads.emplace_back([&](){
			while (true) {
				done.arrive_and_wait();
				if (ex) return;
				int ti = left.fetch_sub(1);
				if (ti>0 && f) f(ti-1);
				done.arrive_and_wait();
			}
		});
	}

	void launch(function<void(int)> nf, int nthread) {
		if (f) throw runtime_error("pool in use");
		if (nthread>threads.size()) throw runtime_error("can't launch that many threads");
		f=nf, left.store(nthread);
		done.arrive_and_wait();
	}

	void join() {
		done.arrive_and_wait();
		f=function<void(int)>();
	}

	~Pool() {
		if (f) done.arrive_and_wait();
		ex=true;
		done.arrive_and_wait();
		for (auto& t: threads) t.join();
	}
};

struct State {
	vector<bool> solution;
	QUBO const& qubo;
	Settings const& set;

	void anneal(Pool& pool) {
		minstd_rand gen(set.seed);
		uniform_int_distribution<> rand_bool(0,1);

		vector<bool> tmp(qubo.n);
		for (int i=0; i<qubo.n; i++) tmp[i]=rand_bool(gen);
		
		atomic<int> thd;
		float energy=0, t=set.T_0;
		int i=0;

		vector<float> diag_float(qubo.diag.begin(), qubo.diag.end());
		vector<int> adj_i;
		vector<int> adj_j;
		vector<double> adj_float_d;
		for (int i=0; i<qubo.n; i++) {
			adj_i.push_back(adj_j.size());
			for (auto [a,b]: qubo.adj[i]) {
				adj_j.push_back(a);
				adj_float_d.push_back(b);
			}
		}

		adj_i.push_back(adj_j.size());

		int stop=set.stop_threshold;
		bool ex=false;

		float best=0;
		solution=tmp;

		auto run = [
				&set=set, n=qubo.n,
				diag=diag_float.data(), adj_i=adj_i.data(),
				adj_j=adj_j.data(), adj_d=adj_float_d.data(),
				&thd,&t,&ex,&best,
				&i,&energy,&stop,&tmp=tmp,&solution=solution
			] (int thread_i) {

			auto thread_gen = minstd_rand(set.seed^thread_i);
			uniform_int_distribution flip_dis(0, n-1);
			uniform_real_distribution<float> dis;

			float thread_temp_mul = uniform_real_distribution<float>(0.1,10)(thread_gen);
			float thread_t=set.T_0*thread_temp_mul;
			float thread_energy=0;
			vector<bool> sol=tmp;

			int nxt_sync = set.synchronize_interval;
			int nxt_thd = thread_i==set.nthread ? 0 : thread_i+1;

			while (true) {
				if (--nxt_sync==0) {
					nxt_sync = set.synchronize_interval;

					if (thd.load(memory_order_acquire)==thread_i) {
						if (thread_energy<best) {
							best=thread_energy;
							solution=sol;
							if (--stop<=0) ex=true;
						} else {
							stop=set.stop_threshold;
						}

						if (ex) {
							thd.store(nxt_thd, memory_order_release);
							break;
						}

						if (dis(thread_gen) < expf((energy-thread_energy)/t)) {
							energy=thread_energy;
							tmp=sol;
						} else {
							thread_energy=energy;
							sol=tmp;
						}

						if (++i>=set.max_iter) ex=true;
						t*=set.alpha;
						thread_t=t*thread_temp_mul;

						thd.store(nxt_thd, memory_order_release);
					}
				}

				int bit = flip_dis(thread_gen);
				float d=diag[bit];
				for (int i=adj_i[bit]; i<adj_i[bit+1]; i++)
					if (sol[adj_j[i]]) d+=adj_d[i];
				if (sol[bit]) d=-d;
				
				if (dis(thread_gen) < expf(-d/thread_t)) {
					sol[bit]=!sol[bit];
					thread_energy+=d;
				}
			}
		};

		pool.launch(run, set.nthread);
		run(set.nthread);
		pool.join();
	}

	void anneal2(Pool& pool, optional<atomic<int>> total_its=nullopt) {
		minstd_rand gen(set.seed);
		uniform_int_distribution<> rand_bool(0,1);

		vector<bool> tmp(qubo.n);
		for (int i=0; i<qubo.n; i++) tmp[i]=rand_bool(gen);
		
		atomic<int> thd;
		float energy=0, t=set.T_0;
		int i=0;

		vector<float> diag_float(qubo.diag.begin(), qubo.diag.end());
		vector<int> adj_i;
		vector<int> adj_j;
		vector<double> adj_float_d;
		for (int i=0; i<qubo.n; i++) {
			adj_i.push_back(adj_j.size());
			for (auto [a,b]: qubo.adj[i]) {
				adj_j.push_back(a);
				adj_float_d.push_back(b);
			}
		}

		adj_i.push_back(adj_j.size());

		int stop=set.stop_threshold, restart=set.restart_threshold;
		bool ex=false;

		float best=0;
		solution=tmp;

		auto run = [
				&set=set, n=qubo.n,
				diag=diag_float.data(), adj_i=adj_i.data(),
				adj_j=adj_j.data(), adj_d=adj_float_d.data(),
				&thd,&t,&ex,&best,&restart,&total_its,
				&i,&energy,&stop,&tmp=tmp,&solution=solution
			] (int thread_i) {

			auto thread_gen = minstd_rand(set.seed^thread_i);
			uniform_int_distribution flip_dis(0, n-1);
			uniform_real_distribution<float> dis;

			float thread_temp_mul = uniform_real_distribution<float>(0.1,10)(thread_gen);
			float thread_t=set.T_0*thread_temp_mul;
			float thread_energy=0;
			vector<bool> sol=tmp;

			int nxt_sync = set.synchronize_interval;
			int nxt_thd = thread_i==set.nthread ? 0 : thread_i+1;

			while (true) {
				if (--nxt_sync==0) {
					nxt_sync = set.synchronize_interval;

					if (thd.load(memory_order_acquire)==thread_i) {
						if (thread_energy<best) {
							best=thread_energy;
							solution=sol;
							if (--stop<=0) ex=true;
						} else {
							stop=set.stop_threshold;
						}

						if (ex) {
							thd.store(nxt_thd, memory_order_release);
							break;
						}

						if (dis(thread_gen) < expf((energy-thread_energy)/t)) {
							energy=thread_energy;
							tmp=sol;

							restart=set.restart_threshold;
						} else {
							thread_energy=energy;
							sol=tmp;

							if (--restart<=0) {
								t=sqrtf(set.T_0*t);
								// t=set.T_0;
								thread_temp_mul = uniform_real_distribution<float>(0.1,10)(thread_gen);
								restart=set.restart_threshold;
							}
						}

						if (++i>=set.max_iter) ex=true;
						t*=set.alpha;
						thread_t=t*thread_temp_mul;

						thd.store(nxt_thd, memory_order_release);
					}
				}

				int bit = flip_dis(thread_gen);
				float d=diag[bit];
				for (int i=adj_i[bit]; i<adj_i[bit+1]; i++)
					if (sol[adj_j[i]]) d+=adj_d[i];
				if (sol[bit]) d=-d;
				
				if (dis(thread_gen) < expf(-d/thread_t)) {
					sol[bit]=!sol[bit];
					thread_energy+=d;
				}

				if (total_its) total_its->operator++();
			}
		};

		pool.launch(run, set.nthread);
		run(set.nthread);
		pool.join();
	}

	double val() {
		double x=0;
		for (auto [a,b]: qubo.entries)
			if (solution[a[0]]&&solution[a[1]]) x+=b;
		return x;
	}
};

string format_time(double nanos) {
	if (nanos<1e3) return format("{:.3f} ns", nanos);
	if (nanos<1e6) return format("{:.3f} us", nanos/1e3);
	if (nanos<1e9) return format("{:.3f} ms", nanos/1e6);
	return format("{:.3f} s", nanos/1e9);
}

struct BenchmarkResult {
	vector<size_t> nanos;
	double mean, dev, total;
	BenchmarkResult(vector<size_t>&& nanos_): nanos(nanos_) {
		sort(nanos.begin(), nanos.end());
		total=double(reduce(nanos.begin(), nanos.end()));
		mean=total/double(nanos.size());

		for (double x: nanos) dev+=(x-mean)*(x-mean);
		dev/=nanos.size();
		dev = sqrt(dev);
	}

	// statistical summary functions: mean, median, quartiles, stddev
	friend ostream& operator<<(ostream& os, BenchmarkResult const& bench) {
		cout<<bench.nanos.size()<<" trials, "<<format_time(bench.mean)<<" mean, "<<format_time(bench.dev)<<" stddev"<<endl;
		for (size_t i=0; i<=4; i++) {
			cout<<"Q"<<i<<": "<<format_time(bench.nanos[(i*(bench.nanos.size()-1))/4]);
			i==4 ? cout<<endl : cout<<", ";
		}

		return os;
	}

	void plot(ostream& os) {
		os<<"set boxwidth 0.5"<<endl;
		os<<"set style fill solid"<<endl;
		os<<"set term pdfcairo"<<endl;
		os<<"set output \"./bench.pdf\""<<endl;
		os<<"set yrange [0:]"<<endl;
		os<<"set xrange [0:]"<<endl;
		os<<"plot '-' using 1 bins binwidth=10000000 notitle with boxes"<<endl;

		for (size_t i=0; i<nanos.size(); i++) {
			os<<nanos[i]<<endl;
		}

		os<<"e"<<endl;
	}
};

vector<BenchmarkResult> bench(int n, initializer_list<function<void()>> fs) {
	vector<vector<size_t>> res(fs.size());

	while (n--) {
		for (int i=0; i<fs.size(); i++) {
			auto& f = *(fs.begin()+i);
			auto s = chrono::high_resolution_clock::now();
			f();
			auto e = chrono::high_resolution_clock::now();
			size_t ns = duration_cast<chrono::nanoseconds>(e-s).count();
			res[i].push_back(ns);
		}
	}

	vector<BenchmarkResult> out;
	for (auto& x: res) out.emplace_back(std::move(x));
	return out;
}

size_t simple_search(size_t l, size_t orig_r, function<bool(size_t)> f, size_t threshold=100) {
	size_t c_thresh = 1;
	while (c_thresh<threshold) {
		c_thresh=min(2*c_thresh, threshold);
		size_t r=orig_r+1;
		while (l<r) {
			size_t mid = (l+r)/2;
			bool failed=false;
			for (size_t i=0; i<c_thresh; i++) {
				if (!f(mid)) {failed=true; break;}
			}

			if (failed && mid==orig_r) throw runtime_error("param search failed");
			else if (failed) l=mid+1;
			else r=mid;
		}
	}

	return l;
}

struct Parameter {
	int lo, hi, val;
	Parameter(int lo_, int hi_, optional<int> val_=nullopt): lo(lo_), hi(hi_), val(val_ ? *val_ : (lo_+hi_)/2) {}
};

struct OptSettings {
	int iter=400;
	double step=1;
	double threshold=0.1;
	double alpha=0.992;
	double parm_slope_alpha=0.9;
	int warmup=100;
	int seed=random_device{}();
};

//garbage optimizer lmao
void optimize(vector<Parameter>& params, function<double(vector<Parameter> const&)> f, OptSettings const& set=OptSettings()) {
	minstd_rand rng(set.seed);

	double step=set.step;
	vector<double> parm_slope(params.size(),0);

	double cur = f(params);

	for (int it=0; it<set.iter; it++) {
		cout<<"optimize it="<<it<<endl;

		int i=uniform_int_distribution<>(0,params.size()-1)(rng);

		int v2;
		do {
			v2 = uniform_int_distribution<>(params[i].lo,params[i].hi)(rng);
		} while (v2==params[i].val);

		swap(params[i].val,v2);
		double other=f(params);
		swap(params[i].val,v2);

		if (abs(other-cur)>1e-3) {
			double newslope = (other-cur)/double(v2-params[i].val);

			double d = double(params[i].val-v2)/(step*(params[i].hi-params[i].lo));
			double c = 1/(1+d*d);
			parm_slope[i] = c*newslope + set.parm_slope_alpha*parm_slope[i];

			if (abs(parm_slope[i])>set.threshold) {
				double step_c = step*min(double(it+1)/set.warmup,1.0);
				if (parm_slope[i]>0) params[i].val+=floor(step_c*(params[i].hi-params[i].val));
				else params[i].val-=floor(step_c*(params[i].val-params[i].lo));

				cur=f(params);
			}

			cout<<"slope of parameter "<<i<<" is "<<parm_slope[i]<<" (new "<<newslope<<")"<<endl;
		}

		step *= set.alpha;
	}
}

int main() {
	// double v=-38.9281;
	// vector<vector<double>> x = {
	// 	{-17, 10, 10, 10, 0, 20},
	// 	{10, -18, 10, 10, 10, 20},
	// 	{10, 10, -29, 10, 20, 20},
	// 	{10, 10, 10, -19, 10, 10},
	// 	{0, 10, 20, 10, -17, 10},
	// 	{20, 20, 20, 10, 10, -28}
	// };

	// make_triangular(x);
	random_device rd;

	vector<pair<array<int, 2>, double>> rnd_qubo;
	int n=100;
	mt19937 gen(123);
	for (int i=0; i<n*20; i++) {
		int r=uniform_int_distribution<>(0,n-1)(gen);
		int c=uniform_int_distribution<>(0,n-1)(gen);
		if (c>r) swap(r,c);

		rnd_qubo.emplace_back(array<int,2>{r,c}, uniform_real_distribution<>(-10,10)(gen));
	}

	// auto parsed = parse_qubo(read_file("qubo.txt"));

	Pool p;
	Settings default_settings = {
		.max_iter = 8000,
		.nthread=int(p.threads.size()),
		.synchronize_interval=30,
		.stop_threshold=2000, .restart_threshold=1000,
		.T_0 = 100.0, .alpha=1-2e-3,
		.seed = rd()
	};

	QUBO qubo = QUBO(vector(rnd_qubo));

	State optimal {.qubo=qubo, .set=default_settings};
	optimal.anneal(p);
	double v = optimal.val();

	vector<Parameter> parms = {
		Parameter(1000,8000),
		Parameter(5,35),
		Parameter(100,8000),
		Parameter(100,8000),
		Parameter(10,1000),
		Parameter(10,100)
	};

	auto parms_to_settings = [&]() {
		return Settings {
			.max_iter = parms[0].val,
			.nthread=int(p.threads.size()),
			.synchronize_interval=parms[1].val,
			.stop_threshold=parms[2].val, .restart_threshold=parms[3].val,
			.T_0 = float(parms[4].val), .alpha=1.0f-powf(10,-4*float(parms[5].val)/100),
			.seed = rd()
		};
	};

	auto parm_score = [&]() {
		int diff=0;
		Settings s = parms_to_settings();
		for (int i=0; i<7; i++) {
			State state {.qubo=qubo, .set=s};
			state.anneal2(p);
			for (int j=0; j<qubo.n; j++)
				if (state.solution[j]!=optimal.solution[j]) diff++;
		}

		cout<<"current: "<<s<<", diff "<<diff<<endl;

		return -diff*100.0/qubo.n - double(s.max_iter*s.synchronize_interval/1e3);
	};

	optimize(parms, [&](vector<Parameter> const& p){return parm_score();});

	auto set = parms_to_settings();
	cout<<set<<endl<<"testing"<<endl;

	int w1=0, w2=0;
	auto mark = bench(100, {[&](){
		State state {.qubo=qubo, .set=set};
		state.anneal(p);

		if (state.val()-v>1e-3) w1++;
		else if (state.val()<v-1e-3) cout<<"this is terrible, everything is wrong"<<endl;
	}, [&](){
		State state {.qubo=qubo, .set=set};
		state.anneal2(p);

		if (state.val()-v>1e-3) w2++;
		else if (state.val()<v-1e-3) cout<<"this is terrible, everything is wrong"<<endl;
	}});

	cout<<"WA (1): "<<w1<<endl;
	cout<<"WA (2): "<<w2<<endl;
	cout<<"1: "<<mark[0]<<endl;
	cout<<"2: "<<mark[1]<<endl;

	// ofstream plot("./plot.gp");
	// mine2.plot(plot);

	// auto ishan = bench(200, [v,&parsed,&p](){
	// 	double x = old::solve(parsed);
	// 	if (abs(x-v)>1e-3) cerr<<"WA (ishan), got "<<x<<endl;
	// });

	// cout<<"ishan:"<<endl<<ishan<<endl;

	// cout << "Energy: " << state.val() << endl;
	// cout << "Solution: " << state.solution << endl;

	return 0;
}
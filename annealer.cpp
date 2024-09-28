#include <atomic>
#include <chrono>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <random>
#include <algorithm>
#include <format>
#include <functional>
#include <ranges>

#include "annealer_old.hpp"
#include "pool.hpp"
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

	QUBO(vector<vector<double>> const& dense): QUBO(dense_to_sparse(dense)) {}

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
	float T_0, restart_mul, alpha;
	unsigned seed;

	//print settings
	friend ostream& operator<<(ostream& os, Settings const& s) {
		return os<<format(
			"max_iter={}, nthread={}, synchronize_interval={}, stop_threshold={}, restart_threshold={}, T_0={}, restart_mul={}, alpha={}, seed={}",
			s.max_iter, s.nthread, s.synchronize_interval, s.stop_threshold, s.restart_threshold, s.T_0, s.restart_mul, s.alpha, s.seed
		);
	}
};

ostream& operator<<(ostream& os, const vector<bool>& x) {
	for (auto xi : x) os << xi << ' ';
	return os;
}

struct State {
	vector<bool> solution;
	QUBO const& qubo;
	Settings const& set;

	void anneal(Pool& pool) {
		minstd_rand gen(set.seed);
		uniform_int_distribution<> rand_bool(0,1);

		vector<char> tmp(qubo.n,0);
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
		vector<char> out=tmp;
		vector<char> orig=tmp;

		auto run = [
				&set=set, n=qubo.n,
				diag=diag_float.data(), adj_i=adj_i.data(),
				adj_j=adj_j.data(), adj_d=adj_float_d.data(),
				&thd,&t,&ex,&best,&restart,
				&i,&energy,&stop,&tmp,&out,&orig
			] (int thread_i) {

			auto thread_gen = minstd_rand(set.seed^thread_i);
			uniform_int_distribution flip_dis(0, n-1);
			uniform_real_distribution<float> dis;

			float thread_temp_mul = uniform_real_distribution<float>(0.1,10)(thread_gen);
			float thread_t=set.T_0*thread_temp_mul;
			float thread_energy=0;
			vector<char> sol=orig;

			int nxt_sync = set.synchronize_interval;
			int nxt_thd = thread_i==set.nthread ? 0 : thread_i+1;

			while (true) {
				if (--nxt_sync==0) {
					nxt_sync = set.synchronize_interval;

					if (thd.load(memory_order_acquire)==thread_i) {
						if (thread_energy<best) {
							best=thread_energy-1e-7;
							out=sol;
							stop=set.stop_threshold;
						} else if (--stop<=0) {
							ex=true;
						}

						if (ex) {
							thd.store(nxt_thd, memory_order_release);
							break;
						}

						float d = (energy-thread_energy)/t;
						if (d > -10 && (d>0 || dis(thread_gen) < 1.0/(d-1)/(d-1))) {
							energy=thread_energy;
							tmp=sol;

							restart=set.restart_threshold;
						} else {
							thread_energy=energy;
							sol=tmp;

							if (--restart<=0) {
								// t=sqrtf(set.T_0*t);
								t*=set.restart_mul;
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
				if (!sol[bit]) d=-d;
				
				d/=thread_t;
				if (d > -10 && (d>0 || dis(thread_gen) < 1.0/(d-1)/(d-1))) {
					sol[bit]=!sol[bit];
					thread_energy-=d;
				}
			}
		};

		pool.launch(run, set.nthread);
		run(set.nthread);
		pool.join();

		solution.assign(tmp.begin(), tmp.end());
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
	int iter=50;
	double alpha=0.93;
	double alpha2=0.5;
	double step=3;
	int seed=random_device{}();
};

//garbage optimizer lmao
void optimize(vector<Parameter>& params, function<double(vector<Parameter> const&)> f, OptSettings const& set=OptSettings()) {
	minstd_rand rng(set.seed);

	double step=set.step;
	int n=params.size();
	double cur = 0;
	double nc=0;
	for (int it=0; it<set.iter; it++) {
		cur*=nc, cur+=f(params), nc++, cur/=nc;

		normal_distribution<double> dist;
		vector<int> old(n);
		for (int i=0; i<n; i++) {
			old[i]=params[i].val;
			params[i].val=clamp(int(round(params[i].val+step*dist(rng)*(params[i].hi-params[i].lo))), params[i].lo,params[i].hi);
		}

		double v = f(params);
		cout<<"opt "<<it<<": "<<v<<" vs "<<cur<<endl;
		if (v<=1.2*cur) for (int i=0; i<n; i++) params[i].val=old[i];
		else cur=set.alpha2*v + (1-set.alpha2)*cur, nc=set.alpha2 + (1-set.alpha2)*nc;

		step*=set.alpha;
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

	auto parsed = parse_qubo(read_file("qubo.txt"));

	Pool p;
	Settings default_settings = {
		.max_iter = 2000,
		.nthread=int(p.threads.size()),
		.synchronize_interval=10,
		.stop_threshold=500, .restart_threshold=320,
		.T_0 = 50.0, .restart_mul=2.3, .alpha=1-1e-2,
		.seed = rd()
	};

	QUBO qubo = QUBO(vector(parsed));

	State optimal {.qubo=qubo, .set=default_settings};
	optimal.anneal(p);
	double v = optimal.val();
	// cout<<"qubo: "<<qubo<<endl;
	cout<<"v: "<<v<<endl;
	cout<<"sol: "<<optimal.solution<<endl;

	// vector<Parameter> parms = {
	// 	Parameter(1000,8000),
	// 	Parameter(5,35),
	// 	Parameter(100,8000),
	// 	Parameter(100,8000),
	// 	Parameter(10,1000),
	// 	Parameter(10,100)
	// };

	// auto parms_to_settings = [&]() {
	// 	return Settings {
	// 		.max_iter = parms[0].val,
	// 		.nthread=int(p.threads.size()),
	// 		.synchronize_interval=parms[1].val,
	// 		.stop_threshold=parms[2].val, .restart_threshold=parms[3].val,
	// 		.T_0 = float(parms[4].val), .alpha=1.0f-powf(10,-4*float(parms[5].val)/100),
	// 		.seed = rd()
	// 	};
	// };

	// auto parm_score = [&]() {
	// 	double diff=0;
	// 	Settings s = parms_to_settings();
	// 	atomic<int> total_its(0);
	// 	for (int i=0; i<30; i++) {
	// 		State state {.qubo=qubo, .set=s};
	// 		state.anneal2(p, &total_its);
	// 		double d=state.val()-optimal.val();
	// 		diff=max(diff,d*d);
	// 	}

	// 	cout<<"current: "<<s<<", diff "<<diff<<", its: "<<total_its.load()<<endl;

	// 	return -diff*100.0 - double(total_its.load()/1e6);
	// };

	// optimize(parms, [&](vector<Parameter> const& p){return parm_score();});

	// auto set = parms_to_settings();
	auto set = default_settings;
	cout<<set<<endl<<"testing"<<endl;

	int w1=0, w2=0;
	size_t my_its=0, ishan_its=0;
	auto mark = bench(300, {[&](){
		State state {.qubo=qubo, .set=set};
		state.anneal(p);

		if (state.val()-v>1e-3) w1++;
		else if (state.val()<v-1e-3) cout<<"this is terrible, everything is wrong"<<endl;
	}, [&]() {
		double val = old::solve(parsed, p);
		if (val-v>1e-3) w2++;
		else if (val<v-1e-3) cout<<"this is terrible, everything is wrong"<<endl;
	}});

	cout<<"WA (1): "<<w1<<endl;
	cout<<"WA (2): "<<w2<<endl;
	cout<<"1: "<<mark[0]<<endl;
	cout<<"2: "<<mark[1]<<endl;

	cout<<"my its: "<<my_its<<", ishan its: "<<ishan_its<<endl;

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
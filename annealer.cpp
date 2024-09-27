#include <atomic>
#include <chrono>
#include <iostream>
#include <concepts>
#include <limits>
#include <mutex>
#include <ostream>
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

template<class T>
concept Scheduler = requires(T a, double T_0, size_t iter, size_t max_iter) {
	{a(T_0,iter,max_iter)}->std::convertible_to<double>;
};

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
	size_t max_iter, nthread, synchronize_interval, stop_threshold;
	double T_0;
	unsigned seed;
};

ostream& operator << (ostream& os, const vector<bool>& x) {
	for (auto xi : x) os << xi << ' ';
	return os;
}

struct State {
	vector<bool> solution;
	QUBO const& qubo;
	Settings const& set;

	template<Scheduler S>
	void anneal(S s) {
		minstd_rand gen(set.seed);
		uniform_int_distribution<> rand_bool(0,1);

		solution.resize(qubo.n);
		for (int i=0; i<qubo.n; i++) solution[i]=rand_bool(gen);
		
		vector<thread> thrds;

		mutex lock;
		double energy=0, t=set.T_0;
		size_t stop=set.stop_threshold;
		size_t i=0, until_step=set.nthread+1;

		auto run = [&](int thread_i){
			auto thread_gen = minstd_rand(set.seed^thread_i);
			double thread_energy=numeric_limits<double>::infinity();
			uniform_int_distribution flip_dis(0, qubo.n-1);
			uniform_real_distribution dis;

			vector<bool> sol=solution;
			double thread_t=t;
			int nxt_sync = set.synchronize_interval;

			while (true) {
				if (--nxt_sync == 0) {
					nxt_sync=set.synchronize_interval;

					lock_guard guard(lock);

					if (--until_step==0) {
						until_step=set.nthread;
						t=s(set.T_0, ++i, set.max_iter);
					}

					if (i>=set.max_iter || stop==0) break;
					thread_t=t;

					if (dis(thread_gen) < exp((energy-thread_energy)/t)) {
						energy=thread_energy;
						solution=sol;
						stop=set.stop_threshold;
					} else {
						thread_energy=energy;
						sol=solution;
						if (--stop==0) break;
					}
				}

				int bit = flip_dis(thread_gen);
				double d = qubo.diff(bit,sol);
				if (dis(thread_gen) < exp(-d/thread_t)) {
					sol[bit]=!sol[bit];
					thread_energy+=d;
				}
			}
		};

		for (int i=0; i<set.nthread; i++) {
			thrds.emplace_back(run, i);
		}

		run(set.nthread);
		for (auto& t: thrds) t.join();
	}

	template<Scheduler S>
	void anneal2(S s) {
		minstd_rand gen(set.seed);
		uniform_int_distribution<> rand_bool(0,1);

		solution.resize(qubo.n);
		for (int i=0; i<qubo.n; i++) solution[i]=rand_bool(gen);
		
		vector<thread> thrds;

		atomic<size_t> thd;
		float energy=0, t=set.T_0;
		size_t i=0;

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

		auto run = [
				set=set, n=qubo.n,
				diag=diag_float.data(), adj_i=adj_i.data(),
				adj_j=adj_j.data(), adj_d=adj_float_d.data(),
				sol=solution, &thd,&t,&s,
				&i,&energy,&solution=solution
			] (int thread_i) mutable {

			auto thread_gen = minstd_rand(set.seed^thread_i);
			uniform_int_distribution flip_dis(0, n-1);
			uniform_real_distribution<float> dis;

			float thread_temp_mul = uniform_real_distribution<float>(0.1,10)(thread_gen);
			float thread_t=set.T_0*thread_temp_mul;
			float thread_energy=0;

			int nxt_sync = set.synchronize_interval;
			size_t nxt_thd = thread_i==set.nthread ? 0 : thread_i+1;

			while (true) {
				if (--nxt_sync==0) {
					nxt_sync = set.synchronize_interval;

					if (thd.load(memory_order_acquire)==thread_i) {
						if (dis(thread_gen) < expf((energy-thread_energy)/t)) {
							energy=thread_energy;
							solution=sol;
						} else {
							thread_energy=energy;
							sol=solution;
						}

						thd.store(nxt_thd, memory_order_release);
						if (i>=set.max_iter) break;
						t=s(set.T_0, ++i, set.max_iter);
						thread_t=t*thread_temp_mul;
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

		for (int i=0; i<set.nthread; i++) {
			thrds.emplace_back(run, i);
		}

		run(set.nthread);
		for (auto& t: thrds) t.join();
	}

	double val() {
		double x=0;
		for (auto [a,b]: qubo.entries)
			if (solution[a[0]]&&solution[a[1]]) x+=b;
		return x;
	}
};

double linear_scheduler(double T_0, int iter, int max_iter) {
	return T_0 - (T_0 / max_iter) * iter;
}

Scheduler auto make_geometric_scheduler(double beta) {
	double alpha=1-beta;
	return [alpha, v=double(0), init=false](double T_0, int iter, int max_iter) mutable {
		if (!init) v=T_0, init=true;
		return v*=alpha;
	};
}

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
};

BenchmarkResult bench(int n, function<void()> f) {
	vector<size_t> res;

	while (n--) {
		auto s = chrono::high_resolution_clock::now();
		f();
		auto e = chrono::high_resolution_clock::now();
		size_t ns = duration_cast<chrono::nanoseconds>(e-s).count();
		res.push_back(ns);
	}

	return BenchmarkResult(std::move(res));
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

int main() {
	double v=-38.9281;
	// vector<vector<double>> x = {
	// 	{-17, 10, 10, 10, 0, 20},
	// 	{10, -18, 10, 10, 10, 20},
	// 	{10, 10, -29, 10, 20, 20},
	// 	{10, 10, 10, -19, 10, 10},
	// 	{0, 10, 20, 10, -17, 10},
	// 	{20, 20, 20, 10, 10, -28}
	// };

	// make_triangular(x);

	auto parsed = parse_qubo(read_file("qubo.txt"));

	random_device rd;
	Settings s = {
		.max_iter = 1000,
		.nthread=std::thread::hardware_concurrency()-1,
		.synchronize_interval=20,
		.stop_threshold=100,
		.T_0 = 100.0,
		.seed = rd()
	};

	auto mine = bench(1000, [v,&parsed,&s](){
		State state {.qubo=QUBO(vector(parsed)), .set=s};
		state.anneal(make_geometric_scheduler(1e-2));

		if (abs(state.val()-v)>1e-3) {
			cerr<<"WA (mine), got "<<state.val()<<endl;
		}
	});

	cout<<"mine:"<<endl<<mine<<endl;

	auto mine2 = bench(1000, [v,&parsed,&s](){
		State state {.qubo=QUBO(vector(parsed)), .set=s};
		state.anneal2(make_geometric_scheduler(1e-2));

		if (abs(state.val()-v)>1e-3) {
			cerr<<"WA (v2), got "<<state.val()<<endl;
		}
	});

	cout<<"mine (2):"<<endl<<mine2<<endl;

	auto ishan = bench(200, [v,&parsed](){
		double x = old::solve(parsed);
		if (abs(x-v)>1e-3) cerr<<"WA (ishan), got "<<x<<endl;
	});

	cout<<"ishan:"<<endl<<ishan<<endl;

	// cout << "Energy: " << state.val() << endl;
	// cout << "Solution: " << state.solution << endl;

	return 0;
}
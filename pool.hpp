#pragma once

#include <functional>
#include <thread>
#include <barrier>
#include <vector>

struct Pool {
	std::vector<std::thread> threads;
	std::barrier<> done;
	std::function<void(int)> f;
	std::atomic<int> left;
	bool ex=false;

	Pool(int nthread=std::thread::hardware_concurrency()-1): done(nthread+1) {
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

	void launch(std::function<void(int)> nf, int nthread) {
		if (f) throw std::runtime_error("pool in use");
		if (nthread>threads.size()) throw std::runtime_error("can't launch that many threads");
		f=nf, left.store(nthread);
		done.arrive_and_wait();
	}

	void join() {
		done.arrive_and_wait();
		f=std::function<void(int)>();
	}

	~Pool() {
		if (f) done.arrive_and_wait();
		ex=true;
		done.arrive_and_wait();
		for (auto& t: threads) t.join();
	}
};

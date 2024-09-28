#pragma once
#include "pool.hpp"
#include <array>
#include <vector>

namespace old {
	double solve(std::vector<std::pair<std::array<int, 2>, double>> const& parsed, Pool& p);
}
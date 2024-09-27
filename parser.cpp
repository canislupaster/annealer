#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include "parser.hh"

using namespace std;

vector<pair<array<int, 2>, double>> parse_qubo(const string& input) {
	vector<pair<array<int, 2>, double>> result;

	string cleanInput = input.substr(1, input.size() - 2); // Strip the outermost brackets
	stringstream ss(cleanInput);
	string pairEntry;

	while (getline(ss, pairEntry, '[')) {
		if (pairEntry.empty()) continue;

		size_t closingBracketPos = pairEntry.find(']');
		if (closingBracketPos == string::npos) continue;
		
		int x, y;
		double value;
		char foo; // to consume random characters in the input
		stringstream s(pairEntry);

		// extras: comma, close bracket, comma.
		s >> x >> foo >> y >> foo >> foo >> value;
		result.emplace_back(array<int,2>{x, y}, value);
	}
	return result;
}

string read_file(const string& filename) {
	ifstream file(filename);
	if (!file.is_open()) 
		throw runtime_error(format("could not open file {}", filename));
	stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}
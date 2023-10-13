#include "model.h"


#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <utility>

std::unordered_map<std::string, int> getWordFrequencies(const std::vector<std::string>& docs) {
    std::unordered_map<std::string, int> word_frequencies;

    for (const std::string& doc : docs) {
        std::string word;
        for (char ch : doc) {
            if (isalpha(ch) || ch == '-') {  // Assuming words can contain hyphens
                word += ch;
            } else if (!word.empty()) {
                ++word_frequencies[word];
                word.clear();
            }
        }
        if (!word.empty()) {
            ++word_frequencies[word];
        }
    }

    return word_frequencies;
}

std::vector<std::pair<std::string, int>> getTopNWords(const std::vector<std::string>& docs, int N) {
    std::unordered_map<std::string, int> word_frequencies = getWordFrequencies(docs);

    std::vector<std::pair<std::string, int>> word_freq_vec(word_frequencies.begin(), word_frequencies.end());
    std::sort(word_freq_vec.begin(), word_freq_vec.end(),
              [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                  return b.second < a.second;  // Sort in descending order of frequency
              });

    word_freq_vec.resize(std::min(N, static_cast<int>(word_freq_vec.size())));  // Resize to only include top N words

    return word_freq_vec;
}

int main() {
	ProgrammingLanguageClassifier pgcl("data/languages.txt");
	
	clock_t start, end;
  	double cpu_time_used;

  	start = clock();
	pgcl.loadData("../../data1/files", {"train", "valid"});
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Data loading took %f sec to execute \n", cpu_time_used);

    int N = 500;
    std::vector<std::pair<std::string, int>> top_n_words = getTopNWords(pgcl.get_docs(), N);

    for (const auto& pair : top_n_words) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    return 0;
}

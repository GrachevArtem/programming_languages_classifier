// Define these to print extra informational output and warnings.
#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <mlpack.hpp>
#include <filesystem>
#include <wchar.h>
#include <regex>

using namespace std;
using namespace mlpack;
using namespace mlpack::data;

namespace fs = std::filesystem;

using DictionaryType = StringEncodingDictionary<MLPACK_STRING_VIEW>;

class ProgrammingLanguageClassifier 
{	
	private:
		map<string, string> langToExtensionMap;  // Mapping from language to file extension
		SoftmaxRegression model;
		vector<string> docs;
		set<string> uniqueTokens;
		//arma::u64_rowvec labels;
		arma::Row<unsigned long int> labels;
		map<string, int> labelToClassIndexMap;  // Mapping from language to class index
		TfIdfEncoding<SplitByAnyOf::TokenType> encoder = TfIdfEncoding<SplitByAnyOf::TokenType>(TfIdfEncodingPolicy::TfTypes::TERM_FREQUENCY, true);
		SplitByAnyOf tokenizer = SplitByAnyOf(" ");

		void cleanText(std::string& text) {
			// Handling snake_case
			size_t pos = 0;
			while ((pos = text.find('_', pos)) != std::string::npos) {
				text.replace(pos, 1, " ");  // replace underscore with space
				pos += 1;
			}

			// Handling CamelCase
			std::string result;
			for (size_t i = 0; i < text.size(); ++i) {
				if (isupper(text[i]) && i != 0) {
					result += ' ';
				}
				result += text[i];
			}
			text = result;

			// Insert space before and after each (, [, {
			std::regex special_chars(R"([\(\[\{\+\-\:\;\*\/\?\)\]\}])");
			text = std::regex_replace(text, special_chars, " $& ");

			// Normalize whitespace
			std::regex whitespace(R"(\s+)");
			text = std::regex_replace(text, whitespace, " ");

			transform(text.begin(), text.end(), text.begin(), ::tolower);
		}

		void cleanTextForPrediction(std::string& text,
			DictionaryType const& dictionary,
			SplitByAnyOf const& tokenizer) {
			cleanText(text);

			MLPACK_STRING_VIEW strView(text);
			auto token = tokenizer(strView);
			string processedText = "";
			while (!tokenizer.IsTokenEmpty(token))
			{
				/* MLPACK encoder will expand the dictionary if unknown tokens are present in the prediction - text.
				 * To avoid that, simply remove the unknown ones.*/ 
				if (dictionary.HasToken(token)) {
					processedText.append(" ").append(token);
				} else {
					processedText.append(" ").append("UNK");
				}

				token = tokenizer(strView);
			}
			text = processedText;
		}

		vector<string> cleanTextWithInternalDict(vector<string> docs,
			SplitByAnyOf const& tokenizer) {
			
			vector<string> processedDocs;
			for (const std::string& text : docs) {
				MLPACK_STRING_VIEW strView(text);
				auto token = tokenizer(strView);
				string processedText = "";
				while (!tokenizer.IsTokenEmpty(token))
				{
					/* MLPACK encoder will expand the dictionary if unknown tokens are present in the prediction - text.
					* To avoid that, simply remove the unknown ones.*/ 
					if (uniqueTokens.find(std::string(token)) != uniqueTokens.end()) {
						processedText.append(" ").append(token);
					} else {
						processedText.append(" ").append("UNK");
					}


					token = tokenizer(strView);
				}
				processedDocs.push_back(processedText);
			}
			return processedDocs;
		}

	public:
		ProgrammingLanguageClassifier(string lang_file)
		{
			loadLanguageMap(lang_file);
		}

		const vector<string>& get_docs() const {
			return docs;
		}

		void loadLanguageMap(string path)
        {
            ifstream file(path);
            string line;
            while (getline(file, line))
            {
                istringstream iss(line);
                string language, extension;
                iss >> language >> extension;
                langToExtensionMap[language] = extension;
                int classIndex = labelToClassIndexMap.size();
                labelToClassIndexMap[language] = classIndex;
            }

			for (const auto& labelIndexPair : labelToClassIndexMap) {
				cout << labelIndexPair.first << " " << labelIndexPair.second << endl;
			}
			for (const auto& labelIndexPair : langToExtensionMap) {
				cout << labelIndexPair.first << " " << labelIndexPair.second << endl;
			}
        }

        void loadData(const string& datasetFolder, const vector<string>& dataCategories, bool isTrain = true)
        {
            docs = vector<string>();
            vector<int> docTags = vector<int>();
            
			for(const auto& dataCategoryDir : dataCategories)
			{
				string dataDir = datasetFolder + "/" + dataCategoryDir;
				for (const auto& langEntry : langToExtensionMap)
				{
					string language = langEntry.first;
					string extension = langEntry.second;
					//int chunk_seed = 0
					//const vector<int> chunk_lengths = 4096; 
					for (const auto& textFile : fs::recursive_directory_iterator(dataDir))
					{

						const int chunk_length = 4096; 
						if (textFile.path().extension() == "." + extension)
						{
							ifstream file(textFile.path());
							string content((std::istreambuf_iterator<char>(file)),
											std::istreambuf_iterator<char>());
							file.close();

							size_t position = 0;
							while (position + chunk_length <= content.size()) 
							{
								string chunk = content.substr(position, chunk_length);
								position += chunk_length;

								size_t extend_pos = content.find(" ", position);

								if (extend_pos != string::npos) 
								{
									chunk += content.substr(position, extend_pos - position + 1);
									position = extend_pos + 1;  // Update the position to after the delimiter
								}
								
								if (isTrain) {
									cleanText(chunk);
								} else { 
									cleanTextForPrediction(chunk, encoder.Dictionary(), tokenizer);
								}
								if (chunk == "") {
									continue;
								}
								docs.push_back(chunk);
								docTags.push_back(labelToClassIndexMap[language]);
								
							}

							// Optional: Handle any remaining characters at the end of the file
							if (position < content.size()) 
							{
								string chunk = content.substr(position);
								if (isTrain) {
									cleanText(chunk);
								} else { 
									cleanTextForPrediction(chunk, encoder.Dictionary(), tokenizer);
								}
								if (chunk != "") {
									docs.push_back(chunk);
									docTags.push_back(labelToClassIndexMap[language]);
								}
							}
						}
					}
				}
			}
			if (isTrain) {
				create_internal_dict();
				docs = cleanTextWithInternalDict(docs, tokenizer);
			}
			cout << "Docs size:" << endl;
			cout << docTags.size() << endl;
			cout << docs.size() << endl;
            
            labels = arma::conv_to<arma::Row<unsigned long int>>::from(docTags);
        }

		void train() 
		{	
			arma::mat trainingData;
			encoder.Encode(docs, trainingData, tokenizer);
			const DictionaryType& dictionary = encoder.Dictionary();
			std::cout << "dictionary_size: " << endl;
			std::cout << dictionary.Size() << endl;
			model = SoftmaxRegression(dictionary.Size(), labelToClassIndexMap.size());
			mlpack::ShuffleData(trainingData, labels, trainingData, labels);
			cout << trainingData.n_rows << " " << trainingData.n_cols << endl;
			cout << labels.n_rows << " " << labels.n_cols << endl;
			model.Train(trainingData, labels, labelToClassIndexMap.size());
		}

		void test()
		{
			// 1. Load Testing Data
			// Assuming docs and labels are class members and have been loaded already
			size_t batch_size = 1000;
			// Determine the number of batches
			size_t num_batches = (docs.size() + batch_size - 1) / batch_size;

			size_t correctPredictions = 0;
			size_t totalPredictions = 0;

			for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx)
			{
				// 2. Preprocess and Encode Batch
				size_t start_idx = batch_idx * batch_size;
				size_t end_idx = std::min(start_idx + batch_size, docs.size());
				vector<string> batch_docs(docs.begin() + start_idx, docs.begin() + end_idx);
				arma::mat batch_testData;


				DictionaryType dictionaryBefore = encoder.Dictionary();
				std::unordered_map<std::basic_string_view<char>, size_t> tokensBefore = dictionaryBefore.Mapping();

				std::cout << "Before: " << encoder.Dictionary().Size() << endl;
				encoder.Encode(batch_docs, batch_testData, tokenizer);
				std::cout << "After: " << encoder.Dictionary().Size() << endl;

				DictionaryType dictionaryAfter = encoder.Dictionary();
				std::unordered_map<std::basic_string_view<char>, size_t> tokensAfter = dictionaryAfter.Mapping();

				// Find and print the new tokens
				for (const auto& tokenPair : tokensAfter) {
					if (tokensBefore.find(tokenPair.first) == tokensBefore.end()) {
						std::cout << "New token: " << tokenPair.first << std::endl;
					}
				}
				// 3. Prediction
				arma::Row<size_t> batch_predictions;

				std::cout << "Dataset dims:";
				std::cout << batch_testData.n_rows << " " << batch_testData.n_cols << endl;
				model.Classify(batch_testData, batch_predictions);

				// 4. Evaluation (within batch)
				vector<int> batch_labels(labels.begin() + start_idx, labels.begin() + end_idx);
				for (size_t i = 0; i < batch_predictions.n_elem; ++i)
				{
					totalPredictions++;
					if (static_cast<size_t>(batch_labels[i]) == batch_predictions(i))
					{
						correctPredictions++;
					}
				}
			}

			// Final Evaluation (across all batches)
			double accuracy = static_cast<double>(correctPredictions) / static_cast<double>(totalPredictions);
			std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
		}

		void evaluateAccuracy(const arma::Row<unsigned long int>& trueLabels, const arma::Row<size_t>& predictions)
		{
			// Ensure that the number of true labels matches the number of predictions
			assert(trueLabels.n_elem == predictions.n_elem);

			size_t correctPredictions = 0;

			// Count the number of correct predictions
			for (size_t i = 0; i < trueLabels.size(); ++i)
			{
				if (static_cast<size_t>(trueLabels[i]) == predictions(i))
				{
					++correctPredictions;
				}
			}

			// Compute accuracy
			double accuracy = static_cast<double>(correctPredictions) / static_cast<double>(trueLabels.size());

			std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
		}

		void save_model() 
		{
			mlpack::data::Save("data/model/encoder.bin", "encoder", encoder);
			mlpack::data::Save("data/model/model.bin", "model", model);
		}

		void load_model()
        {
            mlpack::data::Load("data/model/encoder.bin", "encoder", encoder);
            mlpack::data::Load("data/model/model.bin", "model", model);
        }

		void predictDistribution(string text) 
        {
            const DictionaryType& dictionary = encoder.Dictionary();
            cleanTextForPrediction(text, dictionary, tokenizer);
            arma::mat inputSample;
            encoder.Encode({ text }, inputSample, tokenizer);
            arma::mat probabilities;
            model.Classify(inputSample, probabilities);

            std::cout << "\n\nProbabilities\n";
            for (const auto& labelIndexPair : labelToClassIndexMap)
            {
                std::string label = labelIndexPair.first;
                int index = labelIndexPair.second;
                std::cout << label << ": " << (float)probabilities.at(index) << "\n";
				std::cout << index << endl;
            }
        }

        void predictArgmax(string text)
        {
            const DictionaryType& dictionary = encoder.Dictionary();
            cleanTextForPrediction(text, dictionary, tokenizer);
            arma::mat inputSample;
            encoder.Encode({ text }, inputSample, tokenizer);
            arma::Row<size_t> predictions;
            model.Classify(inputSample, predictions);
            
            std::string predictedLabel;
            for (const auto& labelIndexPair : labelToClassIndexMap)
            {
                if (labelIndexPair.second == predictions(0))
                {
                    predictedLabel = labelIndexPair.first;
                    break;
                }
            }
            
            std::cout << "\n\nPredicted Label: " << predictedLabel << endl;
        }

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

		void create_internal_dict() { 
			auto word_freq_vec = getTopNWords(docs, 1000);

			for (const auto& pair : word_freq_vec) {
				uniqueTokens.insert(pair.first);
			}
		}

};



#include <mlpack.hpp>
#include <filesystem>
#include <wchar.h>
#include <regex>

using namespace std;
using namespace mlpack;
using namespace mlpack::data;

namespace fs = std::filesystem;

using DictionaryType = StringEncodingDictionary<MLPACK_STRING_VIEW>;

class MlpackExample 
{	
	private:
		map<string, string> langToExtensionMap;  // Mapping from language to file extension
		SoftmaxRegression model;
		vector<string> docs;
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
				}

				token = tokenizer(strView);
			}
			text = processedText;
		}

	public:
		MlpackExample(string lang_file)
		{
			loadLanguageMap(lang_file);
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

		void loadData_original(string path) 
		{
			docs = vector<string>();
			vector<int> docTags = vector<int>();
			
			for (const auto& dataCategoryDir : fs::directory_iterator(path))
			{
				//const string dataCategory = dataCategoryDir.path().string().substr(dataCategoryDir.path().string().find_last_of(filesystem::path::preferred_separator) + wcslen(&filesystem::path::preferred_separator), dataCategoryDir.path().string().length());
				const std::string dataCategory = dataCategoryDir.path().filename().string();

				
				for (const auto& textFile : fs::directory_iterator(dataCategoryDir.path()))
				{
					fstream file;
					file.open(textFile.path(), ios::in);
					if (file.is_open()) {
						string content;
						while (getline(file, content)) {
							//cout << content << "\n"; 
						}
						cleanText(content);
						docs.push_back(content);
						docTags.push_back(labelToClassIndexMap[dataCategory]);
						file.close();
					}
				}
			}
			
			labels = arma::conv_to<arma::Row<unsigned long int>>::from(docTags);
		}

        void loadData(const string& datasetFolder) 
        {
            docs = vector<string>();
            vector<int> docTags = vector<int>();
            
			for(const auto& dataCategoryDir : {"/train", "/valid", "/test"})
			{
				string dataDir = datasetFolder + dataCategoryDir;
				for (const auto& langEntry : langToExtensionMap)
				{
					string language = langEntry.first;
					string extension = langEntry.second;
					for (const auto& textFile : fs::recursive_directory_iterator(dataDir))
					{
						const int chunk_length = 1024; 
						if (textFile.path().extension() == "." + extension)
						{
							ifstream file(textFile.path());
							string content((std::istreambuf_iterator<char>(file)),
											std::istreambuf_iterator<char>());
							file.close();
							
							cleanText(content);
							
							size_t position = 0;
							while (position + chunk_length <= content.size()) 
							{
								string chunk = content.substr(position, chunk_length);
								docs.push_back(chunk);
								docTags.push_back(labelToClassIndexMap[language]);
								position += chunk_length;
							}

							// Optional: Handle any remaining characters at the end of the file
							if (position < content.size()) 
							{
								string chunk = content.substr(position);
								docs.push_back(chunk);
								docTags.push_back(labelToClassIndexMap[language]);
							}
						}
					}
				}
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

		void save_model() 
		{
			mlpack::data::Save("data/model/encoder.bin", "encoder", encoder);
			mlpack::data::Save("data/model/model.bin", "model", model);
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

};

int main() 
{
	MlpackExample example("data/languages.txt");
	
	clock_t start, end;
  	double cpu_time_used;

  	start = clock();
	example.loadData("data/files");
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Data loading took %f sec to execute \n", cpu_time_used);

	start = clock();
	example.train();
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Training took %f sec to execute \n", cpu_time_used);

	example.predictDistribution("std::cout << \"Hello World\" << std::endl;");
    example.predictArgmax("std::cout << \"Hello World\" << std::endl;");
	example.save_model();
}

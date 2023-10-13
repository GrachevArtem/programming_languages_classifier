#include "model.h"

int main() 
{
	ProgrammingLanguageClassifier pgcl("data/languages.txt");
	
	clock_t start, end;
  	double cpu_time_used;

  	start = clock();
	pgcl.loadData("../../data1/files", {"train", "valid"});
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Data loading took %f sec to execute \n", cpu_time_used);

	start = clock();
	pgcl.train();
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Training took %f sec to execute \n", cpu_time_used);

	pgcl.predictDistribution("std::cout << \"Hello World\" << std::endl;");
    pgcl.predictArgmax("std::cout << \"Hello World\" << std::endl;");
	pgcl.save_model();
}

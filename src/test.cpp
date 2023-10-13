#include "model.h"

int main() 
{
	ProgrammingLanguageClassifier pgcl("data/languages.txt");
	
	clock_t start, end;
  	double cpu_time_used;

    start = clock();
    pgcl.load_model();
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Model loading took %f sec to execute \n", cpu_time_used);


  	start = clock();
	pgcl.loadData("../../data1/files", {"test"}, false);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Data loading took %f sec to execute \n", cpu_time_used);

    start = clock();
	pgcl.test();
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Testing took %f sec to execute \n", cpu_time_used);
}

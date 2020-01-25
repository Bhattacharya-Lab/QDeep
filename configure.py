import os, sys
working_dir = os.getcwd()
with open("QDeep.py", "rt") as input_file:
	with open("QDeep_tmp.py", "wt") as output_file:
		for line in input_file:
			output_file.write(line.replace('change/to/your/current/directory', working_dir + '/'))
os.system('mv QDeep_tmp.py QDeep.py')

with open("QDeep.py", "rt") as input_file:
        with open("QDeep_tmp.py", "wt") as output_file:
                for line in input_file:
                        output_file.write(line.replace('configured = 0', 'configured = 1'))
os.system('mv QDeep_tmp.py QDeep.py')
os.system('chmod a+x ' + working_dir + '/apps/*')

print('\nConfigured successfully!\n')

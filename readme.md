## MuJoCo Experiments

##### Set-up
1. The experiments have been performed on the following packages:
	- openai gym, version = '0.9.3'
	- MuJoCo mjpro131
	- TensorFlow version 1.10.0

2. Some experiments need changes to be made to the openai gym source code. Copy, paste (, and replace if asked) each and every file inside `MuJoCo/gym_files_to_be_merged` to their equivalent locations of your gym installation.

##### Setting up the demonstrator controllers
1. Go to `MuJoCo_Demonstrator` folder and run the script file `generate_demonstrators.sh`.

##### Generating the learners
1. Run the script file `overnight.sh` present in the root folder.

##### Visualizing the results
1. Open `visualization.ipynb` on jupyter notebook and run all cells.  
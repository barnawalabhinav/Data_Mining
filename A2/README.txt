Data Mining Homework 2
Created By - Abhinav Barnawal (2020CS50415), Shreyansh Singh (2020CS10385), Si Siddhanth Raja (2020CS50443)

# Question 1

(i) Bundled Files in the directory Q1:
    - gaston-1.1: contains the gaston implementation
    - pafi-1.0.1: contains the FGS implementation
    - gSpan6: contains the gSpan implementation
    - format.py: python program to change the format of yeast dataset to the required format
    - graph.py: python program to plot the running times
    - script.sh: script to run the all the algorithms for various minSup and generate plot
    - 167.txt_graph: A set of graphs from the yeast dataset present for convinience (can be replaced)

(ii) How to execute:
    - Enter the Q1 directory by running
    - Save the dataset (in the format of yeast dataset) in this directory
    - If the dataset is named "abc.txt_graph" run the script using the command:
        $ bash script.sh abc.txt_graph
    - The script will generate a file called "temp.txt_graph" that contains the modified datasetan,
        and some ".txt" files containig the running times. They will be deleted automatically after generating the plot.
    - The generated plot will automatically be saved as "running_time.png" in the current directoy.


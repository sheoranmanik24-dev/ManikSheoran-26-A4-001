# ManikSheoran-26-A4-001
This is my GIT repository
<br>
author- Manik
<h1>
  DAY1-26/08/26(YESTERDAY)
</h1>
<P>
1. I used ApnaCollege tutorial to set up git and GitHub. I created my first repository and named it as my name followed by my roll no<br>
2.I wrote a sample code on vs code editor and used commands in terminal to upload and save my code to GitHub using vs code terminal<br>
3. I used commands in vs code terminal as follows to;

Initialize your local folder as a Git repository using
<b>git init</b>
<br>
Stage all of your project files using
<b>git add .</b>
<br>
Save your staged files with a descriptive message using
<b>git commit -m "Initial commit"</b>
<br>
Ensure your default branch is named 'main' using
<b>git branch -m main</b>
<br>
Link your local repository to your remote GitHub repository using
<b>git remote add origin PASTE_YOUR_COPIED_URL_HERE</b>
<br>
Upload your code to GitHub using
<b>git push -u origin main</b>
<br>

4.I also edited the README.md file and added my author name to it
<br>
5. learned basic git commands and powershell commands
<br>

  
</P>

<h1>DAY 2 - 27/08/26</h1>
<p>
  <h5>1.started with task 2 in morning-</h5> faced a lot of problems while figuring out how to detect lanes. I used tutorials of ProgrammingKnowledge to learn open cv for beginners and building the lane detection programme. also while facing errors I used ai tools to understand and counter them. Tutorials could not explain everything so I also used the open cv documentation and gigforgigs for the parts of the code I faced issues.It was tricky to find exact pixel coordinates to fit different camera perspectives, after trying multiple times I was able to figure out the lane detection and mark the region of interest but using a common algorithm it was becoming difficult to do this task and apply it correctly to all 10 images.It became even more challenging while handling curved vs straight road. I did my best to get the desired output but unfortunately I couldn't get the desired output for 2-3 images.I also used ai to add comments to my code to make it more understandable.ADDING THIS STATEMENT ON DAY 3 WHILE CHECKING THE CODE ONCE.I noticed that road shadows were breaking lane lines.i found a solution to it to use adaptive thresholding instead of fixed colors but after trying several times my code showed error(I used ai to detect and fix that error but even after that I could not fix it) . WHAT I WANT TO SAY IS THAT I WANTED TO MAKE CHANGES IN CODE TO FIX IT BUT COULDNT SO AT THE END I LEFT IT AS IT IS. <br>
  <h5>2.started with task 3 at night-</h5> I am still working on task 3 and figuring out and learning using tutorials and exploring other resources. but I am halfway through this task and will complete it by today hopefully.While I have written most of the code for task3 there are still some part of code that needs to be approved for desired output for all the images.Today end it here and will review task3 once tomorrow if it requires any changes for correct output<br>
  
</p>

<h1> DAY 3 - 28/08/26</h1>
<P>
  <h5>1.Today I continued working on task 3.</h5> I faced several issues while trying to get the output.the background gray color was getting confused with obstacles so I used HSV saturation threshold S > 40 to separate colored obstacles from gray background. Another problem that I faced was that white squares were getting missed so to detect them separately I used gray>165 threshold.for several other problems such as a single obstacle producing multiple small, overlapping bounding boxes due to color variation that I was facing I used tutorials and ai tools to solve them. I also used the documentation as it was really helpful. I tried my best to do this task however in 1 or 2 images I could not produce 100% correct results<br>
  <h5>2.Right now its around 9 pm and I am currently working on task 4</h5>. It's a challenging task as while creating the safe path using same algorithm for all images . in many of the images the safe path is going outside the ROAD. I tried several times to fix it but the path is attempting to cut straight across the center non-drivable area instead of staying on the continuous circuit.Right now I am using ai and YouTube tutorials to fix this issue. 
</P>

<h1>DAY 4 - 29/08/26</h1>
<P>
  <H5>Its 1 45 A.M. and I am still working on this task</H5> ,was able to fix the earlier problem to an extent by breaking the Loop Into checkpoint. I did  not tell the program to go from the start line straight to the finish line. Instead, placed a series of dot markers (checkpoints) around the middle of the road and tell the program to connect the dots in order using the for loop.It produced the desired results for many images but in some of the images the output is not as we expect it to be.For today I am leaving it till here and will try to get 100 percent correct results by morning , if its possible.
</P>

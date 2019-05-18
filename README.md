# JSFlowerProbML
A Machine Learning based solution to predict the color of a flower petal

# What does this program do?
This program takes in sets of length, width and color of some flower petals 
and it lets the user to input any sets of length and width to get the program's
prediction of the color of the flower petal.

# Inputs
I've created this situation to train the program as follows:

There's a farmer who works on growing flowers. One day he decides to measure 
the lengths and widths of two different colored flower petals in his farm.
The following chart shows his measurements.
The farmer forgot to take color measurement of the last petal and he needs a way 
to predict what color it might be.

color  | B |  R  | B |  R  |  B |  R  | B |  R  |  ?  |
-------------------------------------------------
length | 2 |  3  | 3 | 3.5 |  2 |  4  | 1 | 5.5 | 4.5 |
-------------------------------------------------
width  | 1 | 1.5 | 1 |  .5 | .5 | 1.5 | 1 |  1  |  1  |

B - Blue 
R - Red

The program takes color of the petals as follows;
    0 for Blue
    1 for Red
    
It predicts according to the algorithm whether the color attribute is closer to 1 or 0.
By the number it gets, it outputs its prediction of the color of the flower petal.
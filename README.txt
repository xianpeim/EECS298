Before running this program, please pip install several things:
gym, tflearn, opencv and tensorflow

Just run the cartpole.py to see how it plays this game.
if output is not average score > 195.0 then run couple times since model depends on random input.

please go to https://flappybird.io/ to play flappybird
change window to appropriate size so that program can read the closest obstacles and label as 2 red lines.
program can show picture captured at a square areas 0,110,550,830, basically left-top corner of screen
	in which it reads corps and process the area of [320, 630],[320, 30],[550, 30],[550, 630] to find obstacles
	
the program can sample and collect data but it takes hours.
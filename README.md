# Framework for detecting, classifying and tracking moving objects

The BehaveAI framework converts motion information into false colours that allow both the human annotator and convolutional neural net (CNN) to easily identify patterns of movement. In addition, the framework integrates conventional static information, allowing both motion streams and static streams to be combined (in a similar fashion to the mammalian visual system). Additionally the framework supports hierarchical models (e.g. detect something from it's movement, then work out what exactly it is from conventional static appearance, or vice-versa); and semi-supervised annotation, allowing the annotator to correct errors made by initial models, making for a more efficient and effective training process.

## Prerequisites

You need a python3 environment with OpenCV and Ultralytics (YOLO), plus a few other standard libraries (numpy, etc...)
 
You only need three files - the annotation script, the classification script, and the settings ini file. First, create a working directory, add these three files, and adjust the settings .ini file to your needs (see below). For convenience, also add your video files to a subdirectory here.
 
## Setting Parameters

You need to adjust the BehaveAI_settings.ini file with your settings. Have a read through the descriptions in the file itself to see what each parameter controls. Note that each class needs an associated keyboard hotkey and colour or it'll throw an error - see the existing format. You can chose betwen YOLO versions (e.g. v8 or v11), and different model sizes (n=nano, s=small etc...). You can also specify what proportion of annotations should be automatically allocated to validation (you can mode the files around later if you'd like though).

### Primary and secondary, static and motion classifiers

Primary classifiers (either motion or static) detect and classify objects across each whole frame. You can mix motion and static in whatever combination you want, the aim here is to make detection as easy as possible for the classifier (some things are easy to see with motion, others static).
 
You can optionally then use secondary classifiers (making your model hierarchical). When you specify these, anything found by a primary classifier will be cropped and sent to the secondary classifier. This can allow you to separate the tasks of detection and classification. See the fly and gull examples for a somewhat complex mix. There are more nuanced settings too. You might not want all primary classes to be sent to the secondary classifier, such as a fly in flight (its wings are a blur, so it's not possible to determine the sex, so flying flies are ignored by the secondary classifiers ).

### Motion strategy

Two different user-selectable motion strategies are available; the ‘exponential’ method calculates the absolute difference between the current frame and the previous frames, exponentially smoothing over successive frames to show different temporal ranges in different colour channels. With this mode, a moving object creates a white ‘difference’ image that leaves behind a motion blur that fades from white to blue, to green, to red. Increasing the exponential smoothing weights allows this method to show events further back in time at no extra computational processing or memory costs when running the classifier because there is no need to re-load any previous frames. This mode is better able to convey changes in speed within each frame; accelerating objects will outpace their red tail, creating a blue-to-green streak, while deceleration will allow the red tail to catch up, creating yellow-to-red tails. The ‘sequential’ method uses discrete frames rather than exponential smoothing, with colours coding the differences between the previous 3 frames (white, blue, green and red going back through time respectively), and is suited to classifying movements over this short range of frames and preserve more spatial information from previous frames (e.g. rather than a smooth tail, an animal’s characteristic wing shapes will remain visible over all four frames). The motion false colour is then optionally blended with the luminance channel to provide greater context – combining motion information with static luminance. The exponential weightings, false-colour composition, and degree of luminance blending are all user-adjustable. Frame skipping can also be used to perform measurements over a larger number of frames (longer time-span) at no additional processing costs (e.g. suited to slow-moving objects whose behaviour is more apparent form faster video playback).
 
## Annotating

### Keyboard shortcuts:

| Key | Function |
|----|----|
| User-selected keys (specified in settings file) |  Primary or secondary class |
| Space | Switch between static and motion view |
| Enter | Save current frame's annotations |
| G | Draw a grey box to hide elements |
| U | Undo last annotation (in annotation or grey box mode). Note that right-click also deletes annotations 
| Right | Next frame |
| Left | Previous frame |
| > | Jump forward 10 frames |
| < | Jump backwards 10 frames |
| Escape | Exit annotation |
| # | Toggle auto-annotate (semi-supervised annotation - only if you've made a model) |
| = | Zoom in preview window (+ key) |
| - | Zoom out preview window |A

Run the BehaveAI_annotation.py script and select a video (to run a python script, either call the file from a command line, or use an IDE such as Anaconda or Geany). You can track through and draw boxes over the things you want to classify. Note all the keyboard shortcuts. It has undo functions, grey-out functions, track single or 10-frame jumps etc... plus all the primary and secondary classes. With each frame, make sure you select (or grey out) everything. Press 'enter' to save the frame and move on.
 
Move through your videos, avoiding annotating the same thing over and over again (to avoid over-fitting, focus on variation). Remember that each frame you add to the annotation dataset (by pressing 'enter') must have **everything** annotated that should be classified. e.g. missing something out will confuse the training. You'll likely encounter cases where you're not sure how to classify something. These borderline cases can be important to help training the model, but if you don't want to add it, draw a grey box over the confusing element (e.g. the target is occluded so you can't tell what it's doing). Try to get the annotation selection box neatly to the edges of the target. Remember that the size and shape of the box might need to vary between motion and static modes (e.g. a flying fly takes up a large box in motion mode compared to static due to its rainbow motion tail). 
 
The aim is to build up an annotation dataset for initial model training, so focus on getting a broad range of backgrounds, contexts etc... Also think about those transition points where one behaviour is turning into another (e.g. switching from walking to flying). Have a clear rule-set in your head when annotating so you can tell what every frame should be (remember, if you can't do this, the CNN will struggle too). With a bit of practice the false colour motion tails will tell you exactly what the target was doing over the previous frames and you'll gain confidence in reading these cues.
 
50-100 annotations of each class should be sufficient to run a quick initial model. This can be done in under 20 minutes for a simple, single class.
 
## Building & running the initial model
Once you've got an initial annotation dataset, run the BehaveAI_classify_track.py script. This will automatically start training the models from your initial annotation files. At this stage, with a small annotation set it should be quite fast. This will create an initial model, and you can see that the model and the YOLO performance data are placed into a new directory in your working directory. Have a look at the output and don't worry if it's not performing amazingly at this stage.

## Auto-Annotation

Once you've trained an initial model the annotation script it will use this in a semi-supervised 'auto-annotation' fashion, attempting to detect and classify things as you annotate. This will show you where the model is working well, and where it's not (closing the training loop). Use this opportunity to correct any errors it's making. e.g. add things it missed (false negatives), redraw boundaries, correct classes etc... importantly you can also add plain background frames (with nothing annotated) in cases where it's making false positive errors. Also focus on those borderline cases whether the model isn't confident (it shows you the confidence). Remember to press 'enter' as before to save a new annotation frame. Aim to increase your annotation set - perhaps doubling the size. Now re-run the BehaveAI_classify_track.py script and it will note that you've added more annotations and ask whether you want to re-train the model. Select 'y' (yes) and it will do so. You can repeat this auto-annotation cycle until you achieve the model performance and versatility required, helping to avoid annotating more than necessary, and also avoiding over-fitting.

If you'd prefer to train the model from scratch rather than retrain your existing model (perhaps also switching between different YOLO version and model sizes), simply move or rename the relevant model directories, adjust the ini settings and re-run the script. Old models are kept as backups, so nothing should be lost. Note that you must not change the motion parameters after building our annotation set; annotations are saved only with the current setting and cannot be altered afterwards.

## Batch processing

Once you're happy with the model, create a directory in your working directory called 'input' and put some videos in it (ideally not the ones you used for training in order to be conservative). Run the BehaveAI_classify_track.py script. It will train the models if necessary, and will then run through the videos in your 'input' directory. You might want to play with the Kalman filter parameters to improve the tracking performance - you can adjust them to specify how fast the objects can change direction, and how much underlying noise there is in the positional accuracy.

This will also output a .csv file that details all the detections frame-by-frame, together with confidence, and tracking ID.

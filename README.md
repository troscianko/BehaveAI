<img width="450" height="76" alt="BehaveAi logo" src="https://github.com/user-attachments/assets/73c30b1b-d73c-4f63-8783-13270bc14b8e" />

# A framework for detecting, classifying and tracking moving objects


The BehaveAI framework converts motion information into false colours that allow both the human annotator and fully convolutional neural network (F-CNN) to easily identify patterns of movement. In addition, the framework integrates conventional static information, allowing both motion streams and static streams to be combined (in a similar fashion to the mammalian visual system). The framework also supports hierarchical models (e.g. detect something from it's movement, then work out what exactly it is from conventional static appearance, or vice-versa); and semi-supervised annotation, allowing the annotator to rapidly correct errors made by initial models, making for a more efficient and effective training process.

#### Key features:
- Fast user-friendly annotation - in under an hour you can create a powerful tracking model
- Identifies objects and their behaviour based on motion and/or static appearance
- Identifies small, fast moving, and motion-blurred targets
- Can track any type(s) of animal/object
- Tracks multiple individuals, classifying the behaviour of each independently
- Built around the verstile YOLO (You Only Look Once) architecture
- Computationally efficient - runs fine on low-end devices without GPUs
- Lightweight installation and intuitive user interface
- Free & open source (GNU Afferro General Public License)

## Prerequisites & installation

You need a python3 environment (Windows, Linux or Mac) with a few extra libraries: OpenCV, numpy, ultralytics, scipy, and PyYAML. These can be installed using pip with the following command:

```shell
pip install opencv-python numpy ultralytics scipy PyYAML
```

A CUDA-enabled GPU speeds up the training, but it works fine without.
 
Place the BehaveAI files in a working directory and [adjust the _BehaveAI_settings.ini_ file](#setting-parameters) to fit your needs using a text editor. For convenience, also create a subdirectory here named 'clips' and place your video files within. Run the BehaveAI.py script (either from the command line, Anaconda, or an IDE such as Geany, Jupyter, or Visual Studio Code). This will bring up the launcher GUI:

<img width="600" alt="Launcher GUI" src="https://github.com/user-attachments/assets/f4ee9768-724f-4d79-a1f5-d5c60b7f0d99" />

Click _Annotate_ and select a file for annotation. Once you've done enough annotating click _Train & batch classify_. 
 
## Setting parameters

You need to adjust the BehaveAI_settings.ini file with your settings. Have a read through the [table](#parameters) below to see what each parameter controls. Note that each class needs an associated keyboard hotkey and colour code or it'll throw an error - see the existing format. You can chose between [YOLO versions](https://github.com/ultralytics/ultralytics/tree/main?tab=readme-ov-file#-documentation) (e.g. YOLOv8 or YOLO11), and [model sizes](https://github.com/ultralytics/ultralytics?tab=readme-ov-file#-models) (e.g. n=nano or s=small). You can also specify what proportion of annotations should be automatically allocated to training and validation (you can manually move the files around later if you'd like though - just be sure to move both images and labels between the 'train' and 'val' subdirectories).

### Primary and secondary, static and motion classifiers

The BehaveAI framework uses two streams of video information, still (_static_) frames, and false-colour-motion (_motion_) frames. Within these, [primary classifiers](#simple-tracking-example) (either motion or static) detect and classify objects across entire frames. You must specify at least one primary classifier, but can specify multiple across both motion and static streams. These can encompass both the same target in different motion states (e.g. stationary vs flying bird), and different targets in their respective motion states (e.g. flying bird vs swimming fish). The aim here is to make detection as easy as possible for the classifier (some things are easier to see with motion, others static), while incoperating target types desired by the user.
 
You can then optionally use secondary classifiers to identify featues within primary classes (making your model hierarchical). When you specify these, anything found by a primary classifier will be cropped and sent to the secondary classifier (e.g. male vs female classification for the primary class of a stationary bird). This can allow you to separate the tasks of detection and feature extraction. See the [fly](#complex-hierarchical-example) and [gull](#motion-strategy) examples for a somewhat complex mix. There is nuance to utilising these settings effectively. You might not want all primary classes to be sent to a secondary classifier, such as a fly in flight (its wings are a blur, so it's not possible to determine the sex, so flying flies are ignored by the secondary classifiers). For the highest computational efficiecy, specify a single primary class and let the secondary classifiers (which are much faster as they use cropped regions) do more work.

#### Simple tracking example

Tracking moths flying against a moving background using only the motion stream (_BehaveAI_settings.ini_ file):

```ini
primary_motion_classes = moth_motion
primary_motion_colors = 246,97,81
primary_motion_hotkeys = m
motion_blocks_static = false

secondary_motion_classes = 0
secondary_motion_colors = 0
secondary_motion_hotkeys = 0

primary_static_classes = 0
primary_static_colors = 0
primary_static_hotkeys = 0
static_blocks_motion = false

secondary_static_classes = 0
secondary_static_colors = 0
secondary_static_hotkeys = 0

ignore_secondary = 
save_empty_frames = true
dominant_source = confidence
```

#### Complex hierarchical example:

Tracking flies on lilly pads using three primary motion classes (_walk_, _fly_ and _display_). When the flies aren't moving they're difficult to spot in the motion stream, so we also add a primary static class (_rest_) to find them from the static stream. However, the static classifier will be able to see all the cases of flies walking, flying or displaying that aren't classed as _rest_. This would likely confuse the static classifier because a walking fly looks a lot like a resting fly. So we add _motion_blocks_static = true_ to hide all the instances of walking, flying or displaying flies from the static classifier.

We also want to determine the sex of each fly from its wing markings, so add _male_ and _female_ as secondary static classes. However, whe displaying or in flight these wing marking won't be visible, so we can tell the model to ignore running the secondary classifier for these cases (_ignore_secondary = display, fly_). Finally, flies will often be detected by both the motion and static classifier, but the motion one will be more reliable and make very few false positive errors, so we set this to be the dominant stream for detections (_dominant_source = motion_). This only affects the video output - data from both streams are saved in the output.

_BehaveAI_settings.ini_ file:

```ini
primary_motion_classes = walk, fly, display
primary_motion_colors = 220,138,221; 249,240,107; 246,97,81
primary_motion_hotkeys = w, y, d
motion_blocks_static = true

secondary_motion_classes = 0
secondary_motion_colors = 0
secondary_motion_hotkeys = 0

primary_static_classes = rest
primary_static_colors = 143,193,193
primary_static_hotkeys = r
static_blocks_motion = false

secondary_static_classes = male, female
secondary_static_colors = 153,193,241; 143,240,164
secondary_static_hotkeys = m, f

ignore_secondary = display, fly
save_empty_frames = true
dominant_source = motion
```

### Motion strategy

<img src="https://github.com/user-attachments/assets/46c94f90-5102-466b-91dc-a38d56f3c2dc" width="400"/>

_Figure showing the different motion strategies (exponential vs sequential), plus the function of lum_weight and frame_skip, of a gull taking flight_

Two different user-selectable motion strategies are available; the ‘_exponential_’ method calculates the absolute difference between the current frame and the previous frames, exponentially smoothing over successive frames to show different temporal ranges in different colour channels. With this mode, a moving object creates a white ‘difference’ image that leaves behind a motion blur that fades from white to blue, to green, to red. Increasing the exponential smoothing weights allows this method to show events further back in time at almost no extra computational cost when running the classifier because there is no need to re-load any previous frames. This mode is better able to convey changes in speed within each frame; accelerating objects will outpace their red tail, creating a blue-to-green streak, while deceleration will allow the red tail to catch up, creating yellow-to-red tails.

The ‘_sequential_’ method uses discrete frames rather than exponential smoothing, with colours coding the differences between the previous 3 frames (white, blue, green and red going back through time respectively), and is suited to classifying movements over this short range of frames while preserving more spatial information (e.g. rather than a smooth tail, an animal’s characteristic wing shapes will remain visible over all four frames). The motion false colour is then optionally blended with the luminance channel to provide greater context – combining motion information with static luminance. The exponential weightings, false-colour composition, and degree of luminance blending are all user-adjustable. Frame skipping can also be used to perform measurements over a larger number of frames (representing a longer time-span) at no additional processing costs (e.g. suited to slow-moving objects whose behaviour is more apparent across a longer span of video).

### Parameters
TLDR: The only things you really must change to fit your project are the primary and secondary classes (and each needs keys and colours associated). Have a look at the motion in your examples and consider tweaking the strategy. The defaults for everything else will likely get you started. Other than tracking and Kalman filter settings, you can't change the values mid-way through annotation.

| Parameter | Range | Description |
|----|----|----|
| [...]_classes | Comma-separated list (0=ignore) | List the names of the classifiers (motion & static, primary & secondary) |
| [...]_colors | Comma-separated RGB values, separated with semicolons (0=ignore) | Specify the RGB colours associated with each class |
| [...]_hotkeys | Comma-separated list of single letters (0=ignore) | Specify which keyboard key to associate with each class for annotation |
| motion_blocks_static | _true_ or _false_ | If enabled, this will hide (grey-out) things annotated in the motion stream from the static stream. Useful to avoid confusing the classifier when you're detecting objects from the motion and static streams simultaneously. Check the annotation output images to see how it works, default _true_ |
| static_blocks_motion | _true_ or _false_ | As above, although unlikely to be as useful because objects generally aren't visible to the motion stream when still, default _false_ |
| ignore_secondary | Comma-separated list matching class names | Specify any classes that should be excluded from secondary classification |
| save_empty_frames | _true_ or _false_ | If true, pressing enter saves frames with no annotations |
| dominant_source | _confidence_, _static_, or _motion_ | Specifies which source should be given priority in video output classification (both are saved in the output .csv file) |
| scale_factor | Proportional range | Scales frames in both annotation and classification - values below 1 reduce image size and increase processing speed, but reduce detail, default 1.0 |
| frame_skip | Integers >= 0 | Skips n frames between each processed frame. 0=normal speed, 1=splits every-other-frame. Higher show motion effects further back in time. |
| motion_threshold | Integers 0-255 | Motion below the threshold is eliminated - can reduce noise, though is typically not required |
| line_thickness | Integer >= 1 | Thickness of lines drawn in the GUI and output videos, default 1, 4k displays/video use 2 |
| font_size | Decimal > 0 | Size of font drawn in GUI and output videos, default 0.5, 4k displays/video use |
| val_frequency | Proportion 0-1 | Specifies the probability that any annotation frame will be sent to the validation (rather than training) dataset, suggested range 0.1-0.2 |
| strategy | _exponential_ or _sequential_ | Specifies motion coding, exponential gives smooth motion tails that fade out, sequential shows movement only in the past 3 frames |
| expA & expB | Proportions 0-1 | If using exponential mode, specifies the rate of decay for green (expA) and red (expB) tails respectively. Higher values give slower temporal decay of movement (can see back further). Recommended values 0.5 and 0.8, avoid >0.9, consider using frame_skip to track motion over longer time-periods |
| lum_weight | Proportion 0-1 | Higher values blend the static luminance (grey) frame with the motion colour. Values >0 allow both motion and static information to be combined in the same frame |
| rgb_multipliers | Comma-separated list of 0-255 values | higher values multiply the motion RGB values (making them more saturated), default 4,4,4.
| primary_classifier | Various options from ultralytics | The YOLO model version and size to use. e.g. yolov8s.pt, yolo11s.pt, yolo11n.pt i.e. version 8 or 11, and size (n=nano, s=small, m=medium etc...) note that yolov8 has the 'v' but yolo 11 doesn't. Default yolo11s.pt |
| primary_epochs | Integer | How many training epochs to run, default 50 |
| secondary_classifier | Various options from ultralytics | Similar to above, although secondary classifiers are run from cropped primary classes, so  they use models with the '-cls' suffix. Default yolo11s-cls.pt |
| secondary_epochs | Integer | How many training epochs to run, default 50 |
| [...]_conf_thresh | Proportion 0-1 | The confidence threshold used to label classes in video output, and add them to the .csv data output (which saves the actual confidence too) |
| match_distance_thresh | Integers > 0 | Threshold below which nearby identified object boxes can be combined between frames, default 200, but should be smaller for low-res video, or higher for HD or fast-moving objects |
| delete_after_missed | Integer > 0 | Number of frames after which temporary IDs should be deleted, higher numbers will track objects that disappear and reappear over longer periods |
| centroid_merge_thresh | Integer > 0 | Merges any two boxes within this pixel radius, default 50, but raise for HD video or larger objects |
| iou_thresh | Proportion 0-1 | IOU threshold intersection of two boxes from static vs motion streams, above which any two boxes are combined. Lower numbers will combine objects with less overlap |
| process_noise_pos | Numeric > 0 | Kalman filter - specifies how erratically objects are likely to be moving (increase if objects move more erratically), default 0.01, but experiment with your own videos, values of 1 work with other videos |
| process_noise_vel | Numeric > 0 | Kalman filter - increase if objects change speed/direction frequently, default 0.1 |
| measurement_noise | Numeric > 0 | Kalman filter - specifies the amount of noise in detections (as opposed to true object movement), increase if detections are noisy/jumpy, default 0.1 but likely requires experiment |


## Annotating

<img width="2245" height="1138" alt="image" src="https://github.com/user-attachments/assets/fe5d2d01-6092-44a7-a0e4-0da7cae85dff" />

_Screenshot of the annotation GUI. The main window is showing the current motion frame (and can switch to static). The right-hand bar shows a zoom view of the current cursor position in static (top right) and motion streams (middle right), and also shows a looping animation of the video covering the same time-frame as the user-selected motion settings (bottom right). The bar at the bottom of the screen highlights the available primary (upper case) and secondary (lower case) classes, together with their associated keys. The trackbar at the bottom of the screen allows seeking through the video_

Click _Annotate_ in the laucher or run the BehaveAI_annotation.py script and select a video. You can track through and draw boxes over the things you want to classify. Note all the keyboard shortcuts. It has undo functions, grey-out functions, track single or 10-frame jumps etc... plus all the primary and secondary classes. With each frame, make sure you select (or grey out) all objects listed as a class (otherwise you'll confuse  it). Press 'enter' to save the frame and move on.
 
Move through your videos, avoiding annotating the same thing over and over again (to avoid over-fitting, focus on variation). Remember that each frame you add to the annotation dataset (by pressing 'enter') must have **everything** annotated that should be classified. e.g. missing something out will confuse the training. You'll likely encounter cases where you're not sure how to classify something. These borderline cases can be important to help training the model, but if you don't want to add it, draw a grey box over the confusing element (e.g. the target is occluded so you can't tell what it's doing). Try to get the annotation selection box neatly to the edges of the target. Remember that the size and shape of the box might need to vary between motion and static modes (e.g. a flying fly takes up a large box in motion mode compared to static due to its rainbow motion tail). 

The aim is to build up an annotation dataset for initial model training, so focus on getting a broad range of backgrounds, contexts etc... Also think about those transition points where one behaviour is turning into another (e.g. switching from walking to flying). Have a clear rule-set in your head when annotating so you can tell what every frame should be (remember, if you can't do this, the classifier will struggle too). With a bit of practice the false colour motion tails will tell you exactly what the target was doing over the previous frames and you'll gain confidence in reading these cues.

50-100 annotations of each class should be sufficient to run a quick initial model. This can be done in under 20 minutes for a simple, single class.

### Annotation keyboard shortcuts:

| Key/button | Function |
|----|----|
| Mouse left-click | Draw annotation using the currently selected class |
| Mouse right-click | Delete annotation or grey box |
| User-selected keys (specified in settings file) |  Select the current primary or secondary class |
| Space | Switch between static and motion view |
| Enter | **Save current frame's annotations** - _moving frame before pressing enter will clear any annotations_ |
| Backspace | Clear all annotations in current frame |
| G | Draw a grey box to hide elements from training (e.g. when you're not sure and don't want to confuse the classifier either way) |
| U | Undo last annotation (in annotation or grey box mode). Note that right-click also deletes annotations. You can also search the annotation directories (based on video name and frame number) to manually delete images and labels later if you make mistakes |
| Right arrow | Next frame |
| Left arrow | Previous frame |
| > | Jump forward 10 frames |
| < | Jump backwards 10 frames |
| Escape | Exit annotation |
| # | Toggle auto-annotate (semi-supervised annotation - only if you've made a model) |
| = | Zoom in preview window (+ key) |
| - | Zoom out preview window |


 
## Building & running the initial model
Once you've got an initial annotation dataset (e.g. 50-100 annotations drawn), click _Train & batch classify_ in the launcher or run the BehaveAI_classify_track.py script. This will automatically start training the models from your initial annotation files. The script will also download a YOLO base model, so you need internet access (or pre-download the relevant initial weights file and drop into the working directory). At this stage, with a small annotation set, training should be quite fast even without GPU acceleration (minutes). This will create an initial model, and you can see that the model and the YOLO performance data are placed into a new directory in your working directory. Have a look at the output and don't worry if it's not performing amazingly at this stage.

## Auto-Annotation

Once you've trained an initial model the annotation script it will use this in a semi-supervised 'auto-annotation' fashion, attempting to detect and classify things as you annotate. This will show you where the model is working well, and where it's not (closing the training loop). Use this opportunity to correct any errors it's making. e.g. add things it missed (false negatives), redraw boundaries, correct classes etc... importantly you can also add plain background frames (with nothing annotated) in cases where it's making false positive errors. Also focus on those borderline cases whether the model isn't confident (it shows you the confidence). Remember to press 'enter' as before to save a new annotation frame. Aim to increase your annotation set - perhaps doubling the size. Now re-run the BehaveAI_classify_track.py script and it will note that you've added more annotations and ask whether you want to re-train the model. Select _yes_ and it will do so. You can repeat this auto-annotation cycle until you achieve the model performance and versatility required, helping to avoid annotating more than necessary, and also avoiding over-fitting.

<img width="600" alt="Launcher retrain" src="https://github.com/user-attachments/assets/b075a4ee-b7d6-4624-9ba8-7151144bd080" />

If you'd prefer to train the model from scratch rather than retrain your existing model (perhaps also switching between different YOLO version and model sizes), simply move or rename the relevant model directories, adjust the ini settings if you want to try a different model type or different number of epochs and re-run the script. Old models are renamed as backups, so nothing should be lost. Note that you must not change the motion parameters in the ini file after building our annotation set; annotations are saved only with the current setting and cannot be altered afterwards.

## Batch processing

Once you're happy with the model, create a directory in your working directory called 'input' and put some videos in it (ideally not the ones you used for training in order to be conservative). Run the BehaveAI_classify_track.py script. It will train the models if necessary, and will then run through the videos in your 'input' directory. You might want to play with the Kalman filter parameters to improve the tracking performance - you can adjust them to specify how fast the objects can change direction, and how much underlying noise there is in the positional accuracy.

This will also output a .csv file that details all the detections frame-by-frame, together with confidence, and tracking ID.

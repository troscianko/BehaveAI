<img width="400" height="69" alt="BehaveAI_400" src="https://github.com/user-attachments/assets/6fb5cd16-d266-4e8b-9513-1734a45813bf" />


# A framework for detecting, classifying and tracking moving objects

BehaveAI is a user-friendly tool that identifies and classifies animals and other objects in video footage from their movement as well as their static appearance. The framework converts motion information into false colours that allow both the human annotator and fully convolutional neural network (F-CNN) to easily identify patterns of movement. The framework integrates both motion streams and static streams in a similar fashion to the mammalian visual system, separating the tasks of detection and classification.

The framework also supports hierarchical models (e.g. detect something from it's movement, then work out what exactly it is from conventional static appearance, or vice-versa); and semi-supervised annotation, allowing the annotator to rapidly correct errors made by initial models, making for a more efficient and effective training process.

#### Key features:
- Fast user-friendly annotation - in under an hour you can create a powerful tracking model
- Identifies objects and their behaviour based on motion and/or static appearance
- Identifies small, fast moving, and motion-blurred targets
- Can track any type(s) of animal/object
- Tracks multiple individuals, classifying the behaviour of each independently
- Semi-supervised annotation allows you to learn where the models are weak and focus on improving performance
- Tools for inspecting and editing the annotation library
- Built around the versatile [YOLO](https://github.com/ultralytics/ultralytics) (You Only Look Once) architecture
- Computationally efficient - runs fine on low-end devices without GPUs
- Intuitive user interface with installers for Windows and Linux (including Raspberry Pi)
- Free & open source ([GNU Afferro General Public License](https://github.com/troscianko/BehaveAI/blob/main/LICENSE))

## Manuscript:
[BioRXIV pre-print](https://www.biorxiv.org/content/10.1101/2025.11.04.686536v1)

## Videos:
[<img width="350" alt="Screenshot from 2025-11-04 17-39-02" src="https://github.com/user-attachments/assets/97a6dd4f-b96f-4bea-80ed-5dae832b0891" />](https://www.youtube.com/watch?v=YQG4497kzPY)
[<img width="350" alt="Screenshot from 2025-11-04 17-38-49" src="https://github.com/user-attachments/assets/5d76855e-d24f-4107-a6b9-c13aa98e6f79" />](https://www.youtube.com/watch?v=PiX7Fp2F-Xk)

<iframe width="560" height="315" src="https://www.youtube.com/embed/YQG4497kzPY?si=d-bObuqvwjtvdrzN" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Prerequisites & installation

### Hardware:

A CUDA-enabled GPU speeds up the training significantly, but the framework works fine without.

### Download the scripts:

[Download](https://github.com/troscianko/BehaveAI/archive/refs/heads/main.zip) the BehaveAI files and place them in a working directory on your system. Also make a directory here called 'clips' where you place the videos you'll use for annotation (this is not essential, but placing them here will allow you to alter the motion settings and rebuild the annotation library at a later date). 

### Windows auto install & launch:
Double-click _Windows_Launcher.bat_ and the first time it runs it will set up your python virtual environment and install required libraries. It will attempt to install GPU drivers if they're available, but these vary between system - follow the prompts. Once installed you can just double-click this file again to launch BehaveAI.

### Linux (Ubuntu & Raspbian) auto install & launch:
Right click the _Linux_Launcher.sh_, click 'properties' and enable 'Executable as Program' (or similar), the right-click again and select 'Run as Program' (or similar). On first run this will set up the python virtual environment and install required libraries. Once installed you can just run this script again to launch BehaveAI.

### General installation in any python environment (Windows, Linux & MacOS):
You need a python3 environment with a few extra libraries. Note that you will generally want to use a python 'virtual environment' that keeps your python environment from messing with system libraries. You'll need OpenCV (in linux it's generally best to insall OpenCV system-wide rather than using pip in your virtual environment), numpy, ultralytics, scipy, and PyYAML. These can be installed using pip with the following command:

```shell
pip install opencv-python numpy ultralytics scipy PyYAML
```
 
Run the BehaveAI.py script (either from the command line, [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), or an IDE such as [Geany](https://github.com/geany/geany), [Jupyter](https://github.com/jupyter/notebook), [Spyder](https://github.com/spyder-ide/spyder), or [Visual Studio Code](https://github.com/microsoft/vscode)). This will bring up the launcher GUI:

<img width="600" alt="Launcher GUI" src="https://github.com/user-attachments/assets/c119d32f-fab1-4d47-9cda-5ecdf338d635" />

Click _Annotate_ and select a file for annotation. Once you've done enough annotating click _Train & batch classify_. 
 
## Setting parameters

You need to adjust the _BehaveAI_settings.ini_ [file](#setting-parameters) to fit your needs using a text editor. Have a read through the [table](#parameters) below to see what each parameter controls. Note that each class needs an associated keyboard hotkey and colour code or it'll throw an error - see the existing format. You can chose between [YOLO versions](https://github.com/ultralytics/ultralytics/tree/main?tab=readme-ov-file#-documentation) (e.g. YOLOv8 or YOLO11), and [model sizes](https://github.com/ultralytics/ultralytics?tab=readme-ov-file#-models) (e.g. n=nano or s=small). You can also specify what proportion of annotations should be automatically allocated to training and validation (you can manually move the files around later if you'd like though - just be sure to move both images and labels between the 'train' and 'val' subdirectories).

<p align="center"> <img width="550" alt="BehaveAI structure" src="https://github.com/user-attachments/assets/1e00ffda-7c52-4388-9a9b-fa35c5294f05" /> </p>

_Overview of the BehaveAI framework's static and motion streams, hierarchical model structure, and classifier integration_

### Primary and secondary, static and motion classifiers

The BehaveAI framework uses two streams of video information, still (_static_) frames, and false-colour-motion (_motion_) frames. Within these, [primary classifiers](#simple-tracking-example) (either motion or static) detect and classify objects across entire frames. You must specify at least one primary classifier, but can specify multiple across both motion and static streams. These can encompass both the same target in different motion states (e.g. stationary vs flying bird), and different targets in their respective motion states (e.g. flying bird vs swimming fish). The aim here is to make _detection_ as easy as possible for the classifier (some things are easier to see with motion, others static), while incorporating target types desired by the user.

![Motion Examples 1](https://github.com/user-attachments/assets/fefe09f0-46bc-49f3-b18d-0fa1afdcc48d)

_Example of static and motion frames from the same section of video_
 
You can then optionally use secondary classifiers to identify featues within primary classes (making your model hierarchical). When you specify these, anything found by a primary classifier will be cropped and sent to the secondary classifier (e.g. male vs female classification for the primary class of a stationary bird). This can allow you to separate the tasks of _detection_ and _classification_. See the [fly](#complex-hierarchical-example) and [gull](#motion-strategy) examples for a somewhat complex mix. There is nuance to utilising these settings effectively. You might not want all primary classes to be sent to a secondary classifier, such as a fly in flight (its wings are a blur, so it's not possible to determine the sex, so flying flies are ignored by the secondary classifiers). For the highest computational efficiency, specify a single primary class and let the secondary classifiers (which are much faster as they use cropped regions) do more work.

![Motion Examples 2](https://github.com/user-attachments/assets/d0b1ec20-c306-40c1-a6af-7d72432c47d8)

_Examples of secondary classifiers within the same primary motion class, these being suface and dive within the primary class of swim_

![Motion Examples 3](https://github.com/user-attachments/assets/003ccbb4-9fdf-44b1-9599-4c868dca284c)

_Examples of secondary classifiers within the same primary static class, these being rest and fan within the primary class of perch_

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

#### Complex hierarchical example

Tracking flies on lilly pads using three primary motion classes (_walk_, _fly_ and _display_). When the flies aren't moving they're difficult to spot in the motion stream, so we also add a primary static class (_rest_) to find them from the static stream. However, the static classifier will be able to see all the cases of flies walking, flying or displaying that aren't classed as _rest_. This would likely confuse the static classifier because a walking fly looks a lot like a resting fly. So we add _motion_blocks_static = true_ to hide all the instances of walking, flying or displaying flies from the static classifier.

We also want to determine the sex of each fly from its wing markings, so add _male_ and _female_ as secondary static classes. However, when displaying or in flight these wing marking won't be visible, so we can tell the model to ignore running the secondary classifier for these cases (_ignore_secondary = display, fly_). Finally, flies will often be detected by both the motion and static classifier, but the motion one will be more reliable and make very few false positive errors, so we set this to be the dominant stream for detections (_dominant_source = motion_). This only affects the video output - data from both streams are saved in the output.

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

<img src="https://github.com/user-attachments/assets/60743d69-9e88-4ceb-923c-0b7796f0ce00" width="900"/>


_Examples of the different motion strategies (exponential vs sequential), plus the functions of lum_weight and frame_skip across species_

Two different user-selectable motion strategies are available; the ‘_exponential_’ method calculates the absolute difference between the current frame and the previous frames, exponentially smoothing over successive frames to show different temporal ranges in different colour channels. With this mode, a moving object creates a white ‘difference’ image that leaves behind a motion blur that fades from white to blue, to green, to red. Increasing the exponential smoothing weights allows this method to show events further back in time at almost no extra computational cost when running the classifier because there is no need to re-load any previous frames. This mode is better able to convey changes in speed within each frame; accelerating objects will outpace their red tail, creating a blue-to-green streak, while deceleration will allow the red tail to catch up, creating yellow-to-red tails.

The ‘_sequential_’ method uses discrete frames rather than exponential smoothing, with colours coding the differences between the previous 3 frames (white, blue, green and red going back through time respectively), and is suited to classifying movements over this short range of frames while preserving more spatial information (e.g. rather than a smooth tail, an animal’s characteristic wing shapes will remain visible over all four frames). The motion false colour is then optionally blended with the luminance channel to provide greater context – combining motion information with static luminance. The exponential weightings, false-colour composition, and degree of luminance blending are all user-adjustable. Frame skipping can also be used to perform measurements over a larger number of frames (representing a longer time-span) at no additional processing costs (e.g. suited to slow-moving objects whose behaviour is more apparent across a longer span of video).

In certain cases, the motion stream may not be the best option for classifying activities. An example of this is when the camera itself is moving, meaning that all parts of the image will be subject to motion traces from previous frames, thus obscuring behaviours. As such, it is important to use the annotation interface to assess videos via both the motion and static streams, to determin which is most suitable for classifying behaviours.

![Motion Examples 4](https://github.com/user-attachments/assets/90f0f3ad-dd57-4572-8d1d-eb29f693e82c)

_Example of a case where the static stream may be preferable to the motion stream due to camera movement_

### Parameters
TLDR: The only things you really must change to fit your project are the primary and secondary classes (and each needs keys and colours associated). Have a look at the motion in your examples and consider tweaking the strategy. The defaults for everything else will likely get you started. Note that you can adjust/change any of the motion processing settings at a later date. The framework will notice that you've altered the settings and offer to rebuild the annotation library from the video files using your new settings. This function only works if the videos used for annotations are in the 'clips' directory. You'll also need to re-build the motion models if these settings change (again you'll be prompted to do this if a change is detected). The only thing to be aware of is that the boxes drawn around your moving objects might not be optimised with altered settings. Use the 'Insepct Dataset' function to check the boxes look good.

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
| chromatic_tail_only | 
| expA & expB | Proportion 0-1 | If using exponential mode, specifies the rate of decay for green (expA) and red (expB) tails respectively. Higher values give slower temporal decay of movement (can see back further). Recommended values 0.5 and 0.8, avoid >0.9. For measuring movement further back in time increase _frame_skip_ instead |
| lum_weight | Proportion 0-1 | Higher values blend the static luminance (grey) frame with the motion colour. Values >0 allow both motion and static information to be combined in the same frame | _true_ or _false_ | Default false. If true, it removes the intial white from the difference frame and shows only the colourful tails of the motion track. Useful for preserving more static detail, with motion creating more subtle chromatic tails. Recommend setting _lum_diff_ to 1.0 if enabled. |
| rgb_multipliers | Comma-separated list of 0-255 values | higher values multiply the motion RGB values (making them more saturated), default 4,4,4.
| primary_classifier | Various options from ultralytics | The YOLO model version and size to use. e.g. yolov8s.pt, yolo11s.pt, yolo11n.pt i.e. version 8 or 11, and size (n=nano, s=small, m=medium etc...) note that yolov8 has the 'v' but yolo 11 doesn't. Default yolo11s.pt |
| primary_epochs | Integer | How many training epochs to run, default 50 |
| secondary_classifier | Various options from ultralytics | Similar to above, although secondary classifiers are run from cropped primary classes, so  they use models with the '-cls' suffix. Default yolo11s-cls.pt |
| secondary_epochs | Integer | How many training epochs to run, default 50 |
| use_ncnn | _true_ or _false_ | Default false. If enabled, the YOLO model is automatically converted to an NCNN model, which allows for faster processing on CPU architecture. You can enable after building your model and compare processing speeds with and without this enabled. E.g. results in much faster processing on Raspberry Pis |
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

Click _Annotate_ in the laucher or run the BehaveAI_annotation.py script and select a video. You can track through and draw boxes over the things you want to classify. Note all the [keyboard shortcuts](#annotation-keyboard-shortcuts:). These include undo functions, grey-out functions, track single or 10-frame jumps etc... plus all the primary and secondary classes. With each frame, make sure you select (or grey out) all objects listed as a class (otherwise you'll confuse the model), then press 'enter' to save your annotations and move on.
 
Move through your videos, avoiding annotating the same thing over and over again (to avoid over-fitting, focus on variation). Remember that each frame you add to the annotation dataset (by pressing 'enter') must have **everything** annotated that should be classified. e.g. missing something out will confuse the training as it will be treated as part of the background. You'll likely encounter cases where you're not sure how to classify something. These borderline cases can be important to help training the model, but if you don't want to add them, draw a grey box over the confusing element (e.g. the target is occluded so you can't tell what it's doing). Try to get the annotation selection box neatly to the edges of the target. Remember that the size and shape of the box might need to vary between motion and static modes (e.g. a flying fly takes up a large box in motion mode compared to static due to its rainbow motion tail).

The aim is to build up an annotation dataset for initial model training, so focus on getting a broad range of backgrounds, contexts etc... Also think about those transition points where one behaviour is turning into another (e.g. switching from walking to flying). Have a clear rule-set in your head when annotating so you can tell what every behaviour should be (remember, if you can't do this, the classifier will struggle too). With a bit of practice the false colour motion tails will tell you exactly what the target was doing over the previous frames and you'll gain confidence in reading these cues.

50-100 annotations of each class should be sufficient to run an initial model. This can be done in under 20 minutes for a simple, single class.

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

### Inspect Dataset
Clicking _Inspect Dataset_ from the BehaveAI launcher will open a tool that shows you the labelled/annotated frames in your current dataset. You can use this to check that all your annotations are correct, and delete or adjust them at and stage. The functionality is much the same as the annotation tool above, flipping between motion and static frames. It shows images from the training and validation datasets. Sometimes during annotation you might find that your threshold for discriminating between two classes of something shifts slightly (we learn more as we encounter more borderline cases). This tool lets you go back and undo any errors made early on in the annotation process.

 
## Building & running the initial model

Once you've got an initial annotation dataset (e.g. 50-100 annotations), click _Train & batch classify_ in the launcher or run the BehaveAI_classify_track.py script. This will automatically start training models from your initial annotation files. The script will also download a YOLO base model, so you need internet access (or pre-download the relevant [initial weights file](https://github.com/ultralytics/ultralytics/tree/main?tab=readme-ov-file#-models) and drop it into the working directory). At this stage, with a small annotation set, training should be quite fast even without GPU acceleration (minutes). This will create an initial model, and you can see that the model and the YOLO performance data are placed into a new folder in your working directory. Have a look at the output and don't worry if it's not performing amazingly at this stage.

## Auto-annotation

Once you've trained an initial model the annotation script it will use this in a semi-supervised 'auto-annotation' fashion, attempting to automatically detect and classify things as you annotate. This will show you where the model is working well, and where it's not (closing the training loop). Use this opportunity to correct any errors it's making. Add things that it misses (false negatives), remove any incorrect detections (false positives), redraw boxes that are misaligned, and correct objects that are misclassified. Importantly, you can also add plain background frames (with nothing annotated) in cases where it's falsely detecting elements of the background.

![Annotation Example 1](https://github.com/user-attachments/assets/97ebd2ef-6c6b-4c80-be7d-8b1c4f45e7c4)

_Example of auto-annotation functionality_

Also focus on those borderline cases where the model isn't confident (based on the confidence scores appended to boxes). Remember to press 'enter' as before to save the new annotations on each frame. Aim to increase your annotation set - perhaps doubling the size. Now re-run the BehaveAI_classify_track.py script and it will note that you've added more annotations and ask whether you want to re-train the model. Select _yes_ and it will do so. You can repeat this auto-annotation cycle until you achieve the model performance and versatility required, helping to avoid annotating more than necessary, and also avoiding over-fitting.

<img width="2245" src="https://github.com/troscianko/BehaveAI/blob/Toshea111-patch-1/BehaveAI%20Examples.png" />

_Examples of common errors revealed through auto-annotation_

If you'd prefer to train the model from scratch rather than retrain your existing model (perhaps also switching between different YOLO version or model sizes), simply move or rename the relevant model directories, adjust the BehaveAI_settings.ini file if you want to try a different model type or different number of epochs, and re-run the script. Previous models are renamed as backups, so nothing should be lost. Note that you must not change the motion parameters in the BehaveAI_settings.ini file after building an annotation set; annotations are saved only with the current setting and cannot be altered afterwards.

## Batch processing

Once you're happy with the model, create a folder in your working directory called 'input' containing the videos that you wish to analyse (ideally not the ones you used for training in order to be conservative). Run the BehaveAI_classify_track.py script. It will update the models if necessary, and will then run inference on all of the videos in your 'input' directory, outputting the results as a .csv file that details all the detections frame-by-frame, together with confidence scores, class hierarchies, and tracking IDs. You might want to play with the Kalman filter and confidence threshold parameters to improve tracking performance. You can adjust them to specify how fast objects can change direction, how much underlying noise there is in the positional accuracy, how confident the model needs to be to accept detections, and how much objects can overlap before they are treated as a single entity.

## Workflow

<img width="2245" src="https://github.com/user-attachments/assets/5260ba9f-0392-4d1a-9bb5-0daa8ffbf654" />

_Overview of the BehaveAI pipeline_

Combining the above steps, you can build an efficient workflow for your desired pipeline consisting of manual annotation, training, semi-automated annotation, retraining, and deployment. This process can be repeated as many times as needed with additional video data to deliver a sufficiently accurate model with minimal manual input. Below is a brief example of this when classifying behaviour in ants.

<img width="2245" src="https://github.com/troscianko/BehaveAI/blob/Toshea111-patch-1/Ant%20Cleaning%20Example.gif" />

_Use the GUI to determine how best to use the static and motion streams for annotation_

<img width="2245" src="https://github.com/troscianko/BehaveAI/blob/Toshea111-patch-1/Annotation%20Example.gif" />

_Annotate individuals with the desired behavioural classifiers, switching between static an motion streams as needed_

<img width="2245" src="https://github.com/troscianko/BehaveAI/blob/Toshea111-patch-1/Annotation%20Example%201.gif" />

_Following training of an initial model, check predictions via auto-annotation and correct where required_

<img width="2245" src="https://github.com/troscianko/BehaveAI/blob/Toshea111-patch-1/Deployment%20Example.gif" />

_Once satisfactory performance is achieved, run inference on videos to extract and visualise behavioural metrics_

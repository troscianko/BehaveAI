<img width="400" height="69" alt="BehaveAI_400" src="https://github.com/user-attachments/assets/6fb5cd16-d266-4e8b-9513-1734a45813bf" />


# A framework for detecting, classifying and tracking moving objects

BehaveAI is a user-friendly tool that identifies and classifies animals and other objects in video footage from their movement as well as their static appearance. The framework converts motion information into false colours that allow both the human annotator and convolutional neural network (CNN) to easily identify patterns of movement. The framework integrates both motion streams and static streams in a similar fashion to the mammalian visual system, separating the tasks of detection and classification.

The framework also supports hierarchical models (e.g. detect something from it's movement, then work out what exactly it is from conventional static appearance, or vice-versa); and semi-supervised annotation, allowing the annotator to rapidly correct errors made by initial models, making for a more efficient and effective training process.

![BehaveAI_flies](https://github.com/user-attachments/assets/99ae83fb-c001-4d5a-8338-0607a914d0c4)


#### Key features:
- Identifies objects and their behaviour based on motion and/or static appearance
- Fast user-friendly annotation - in under an hour you can create a powerful tracking model
- Identifies small (2 px!), fast moving, and motion-blurred targets
- Can track any type(s) of animal/object
- Tracks multiple individuals, classifying the behaviour of each independently
- Semi-supervised annotation allows you to learn where the models are weak and focus on improving performance
- Tools for inspecting and editing the annotation library
- Live (edge) support - record videos, train, then run live video processing using the user interface
- Built around the versatile [YOLO](https://github.com/ultralytics/ultralytics) (You Only Look Once) architecture
- Computationally efficient - runs fine on low-end devices without GPUs
- Intuitive user interface with installers for Windows and Linux (including Raspberry Pi) and full user-interface. Also works on MacOS but currently no installer.
- Free & open source ([GNU Afferro General Public License](https://github.com/troscianko/BehaveAI/blob/main/LICENSE))

## User guide & installation instructions

See the project wiki [here](https://github.com/troscianko/BehaveAI/wiki) for detailed user guide and installation instructions.

## Paper & Citation:
[PLOS Biology publication](https://doi.org/10.1371/journal.pbio.3003632)

If you use BehaveAI please cite:

* Troscianko, Jolyon, Thomas A. O’Shea-Wheller, James A. M. Galloway, and Kevin J. Gaston. (2026) 'BehaveAI Enables Rapid Detection and Classification of Objects and Behavior from Motion’. _PLOS Biology_ 24, no. 2 (2026): e3003632. https://doi.org/10.1371/journal.pbio.3003632.


## Video Guide (v1.2):
[<img width="350" alt="Screenshot from 2025-11-04 17-38-49" src="https://github.com/user-attachments/assets/5d76855e-d24f-4107-a6b9-c13aa98e6f79" />](https://www.youtube.com/watch?v=atEL14nxz9s)


See the [project Wiki](https://github.com/troscianko/BehaveAI/wiki) for detailed instructions.

# Classifying electrons using Neural Networks
## Overview
This is the code for training and testing on image data for classification of electrons in a high energy physics (HEP) experiment.
## The Large Hadron Collider
The Large Hadron Collider (LHC) is the biggest and most powerful particle accelerator. It consists of a 27 km ring of superconducting magnets. In the accelerator, two high energy particle beams (proton beams) traveling in nearly the speed of lights are made to collide. Beams are made to go in the opposite directions in separate beam pipes. Beams are guided along the accelerator by a strong magnetic field produced using superconducting electromagnets. The collisions happen at four locations. At the collision points, the energy of particle collisions is transformed into mass resulting in the formation of several particles. Four detectors are placed at these collision points to detect this spray of particles. The biggest of these detectors, CMS and ATLAS, are general purpose detectors. ALICE and LHCb are detectors designed for the search of specific phenomena.
## CMS
Compact Muon Solenoid or CMS is a general-purpose detector designed to observe any new physics phenomena. Though most particles produced by high energy collisions at the LHC are unstable, they decay to stable particles which can be detected by CMS. By identifying these stable particles, their momenta and energy, the detector can recreate a model of the collision. It acts as a high-speed camera, taking 3D photographs of particle collisions.

![cms-event](https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/cmsreleasesn.gif)

## Particle Reconstruction
The interaction of the particle with the different layers of the detector is used to reconstruct the nature and properties of the particle. The silicon tracker tracks particles near the collision region. The calorimeter stops some particles and measures their energy. The magnet bends the particle allowing for momentum measurement. Muon chambers detect muons.

![cms-reconstruction](https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/particle_reco.gif)

This work uses convolutional neural networks to classify electrons using their signature in the calorimeters. Data from the CMS detector is used for this work.

## Electron Classification
Electrons that are away from any hadronic activity (isolated) should be distinguished from those that are close to hadronic activity (non-isolated). Better discrimination gives a better chance of observing new physics signals which mainly has isolated electrons in which case the non-isolated electrons form the background. This work uses the pattern of energy deposits in the calorimeters of the detector around the electron for classification. The pattern of energy deposits is treated as a two-dimensional image where two dimensions are spatial coordinates of the detector and pixel intensities are the amount of energy deposited at different points of the calorimeter.

![img-con](https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/imgcon.png)

In the ideal case, isolated electrons are expected to have a lesser amount of energy deposits around them compared to non-isolated electrons. But factors like pile-up (collisions at the same bunch-crossing) and bremsstrahlung radiation emitted by electrons, makes patterns of both electrons indistinguishable for humans.

Below are some example images of isolated electrons.

<img src="https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/real_ele1.jpg" width="425"/> <img src="https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/real_ele2.jpg" width="425"/>
<img src="https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/real_ele3.jpg" width="425"/> <img src="https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/real_ele4.jpg" width="425"/>

Below are some examples images of non-isolated electrons.

<img src="https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/fake_ele1.jpg" width="425"/> <img src="https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/fake_ele2.jpg" width="425"/>
<img src="https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/fake_ele3.jpg" width="425"/> <img src="https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/fake_ele4.jpg" width="425"/>

Note: The color in this image is only for representation. Actual images used for training and testing of neural networks were grayscale images.

Below is the output histogram of a convolutional neural network after training and testing on 8000 electron images of both categories. (Signal is isolated electrons and background is non-isolated electrons.)

![result](https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/Figure_2-16.png)

The performance of CNN was compared with the performance of traditional physics observable, a parameter called "isolation", used by physicists to classify these electrons. The ROC curve for both is shown below. The CNN outperformed the traditional physics observable for most signal efficiencies (Signal efficiency - True positive rate, Background rejection - True negative rate).

![roc](https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/Figure_2-12.png)

## Instructions

All the information used for creating the images, i.e., the spatial coordinates, pseudorapidity and azimuthal angle values, along with energy deposits at different values of these spatial coordinates, come from CERN open data portal. The simulations which are in AOD format are used to create TTrees. This is done using CMS software (CMSSW). Sample code for this conversion can be found in CMSSW repository. The TTree is then analyzed to select electrons of interest and to create images for each of them. The code for this can be found in FlatTree Analysis repository. Finally, these images become input to a convolutional neural network implemented using Tensorflow. The code for this is ![class_train_test.py](https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/class_train_test.py). A simple neural network is implemented for the same purpose in the script ![ele_class_ann.py](https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/ele_class_ann.py).

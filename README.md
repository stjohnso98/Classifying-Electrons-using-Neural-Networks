# Classifying electrons using Neural Networks
## Overview
This is the code for training and testing on image data for classification of electrons in a high energy physics (HEP) experiment.
## The Large Hadron Collider
The Large Hadron Collider (LHC) is the biggest and most powerful particle accelerator. It consists of a 27 km ring of superconducting magnets. In the accelerator, two high energy particle beams (proton beams) travelling in nearly the speed of lights are made to collide. Beams are made to travel in opposite direction in seperate beam pipes. Beams are guided along the accelarator by a strong magnetic field produced using superconducting electromagnets. The collisions happen at four locations. At the collision points energy of particle collisions are transformed to mass resulting in the formation of several particles. Four detectors are placed at these collision points to detect this spray of particles. The biggest of these detectors, CMS and ATLAS, are general purpose detectors. ALICE and LHCb are detectors designed for search of specific phenomena.
## CMS
Compact Muon Solenoid or CMS is a general purpose detector designed to observe any new physics phenomena. Though most particles produced by high energy collisions at the LHC are unstable, they decay to stable particles which can be detected by CMS. By identifying these stable particles, their momenta and energy, the detector can recreate a model of the collision. It acts as a high speed camera, taking 3D photographs of particle collisions.

![cms-event](https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/cmsreleasesn.gif)

## Particle Reconstruction
The interaction of the particle with the different layers of the detector is used to reconstruct the nature and properties of the particle. The silicon tracker tracks particles near the collision region. The calorimeter stops some particles and measures their energy. The magnet bends the particle allowing for momentum measurement. Muon chambers detect muons.

![cms-reconstruction](https://github.com/stjohnso98/Classifying-Electrons-using-Neural-Networks/blob/master/docs/particle_reco.gif)

This work uses convolutional neural networks to classify electrons using their signature in the calorimeters. Data from CMS detector is used for this work.

## Electron Classification
Electrons that are away from any hadronic activity (isolated) should be distinguished from those that are close to hadronic activity (non-isolated) . Better discrimination gives a better chance of observing new physics signals which mainly has isolated electrons in which case the non-isolated electrons forms the background. This work uses pattern of energy deposits in the calorimeters of the detector around the electron for classification. In the ideal case, isolated electrons is expected to have lesser amount of energy deposits around them compared to non-isolated electrons. But factors like pile-up (collisions at same buch-crossing) and bremsstrahlung radiation emitted by electron, makes patterns of both eletrons indistinguishable for humans.


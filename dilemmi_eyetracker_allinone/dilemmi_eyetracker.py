#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on Tue Nov 21 11:01:21 2023
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# ---- Import Eyetracker ---
import os
import time
import nest_asyncio
import pupil_labs.realtime_api
from pupil_labs.realtime_api.simple import discover_one_device
from datetime import datetime

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'dilemmi'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/daniele/Desktop/file_daniele/eyetracking/psychopy/dilemmi_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=(1024, 768), fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "tutorial" ---
    cross_tutorial = visual.ImageStim(
        win=win,
        name='cross_tutorial', 
        image='images/cross.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    dilemma_1_tutorial = visual.ImageStim(
        win=win,
        name='dilemma_1_tutorial', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    audio_1_tutorial = sound.Sound('A', secs=-1, stereo=True, hamming=True,
        name='audio_1_tutorial')
    audio_1_tutorial.setVolume(1.0)
    cross_2_tutorial = visual.ImageStim(
        win=win,
        name='cross_2_tutorial', 
        image='images/cross.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    dilemma_2_tutorial = visual.ImageStim(
        win=win,
        name='dilemma_2_tutorial', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    audio_2_tutorial = sound.Sound('A', secs=-1, stereo=True, hamming=True,
        name='audio_2_tutorial')
    audio_2_tutorial.setVolume(1.0)
    mouse_tutorial = event.Mouse(win=win)
    x, y = [None, None]
    mouse_tutorial.mouseClock = core.Clock()
    sx_tutorial = visual.ImageStim(
        win=win,
        name='sx_tutorial', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.5, -0.25), size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    dx_tutorial = visual.ImageStim(
        win=win,
        name='dx_tutorial', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.5, -0.25), size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    
    # --- Initialize components for Routine "dilemmi" ---
    cross_img = visual.ImageStim(
        win=win,
        name='cross_img', 
        image='images/cross.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    dilemma_1 = visual.ImageStim(
        win=win,
        name='dilemma_1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    audio_1 = sound.Sound('A', secs=-1, stereo=True, hamming=True,
        name='audio_1')
    audio_1.setVolume(1.0)
    cross_img_2 = visual.ImageStim(
        win=win,
        name='cross_img_2', 
        image='images/cross.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    dilemma_2 = visual.ImageStim(
        win=win,
        name='dilemma_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    audio_2 = sound.Sound('A', secs=-1, stereo=True, hamming=True,
        name='audio_2')
    audio_2.setVolume(1.0)
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    sx = visual.ImageStim(
        win=win,
        name='sx', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.5, -0.25), size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    dx = visual.ImageStim(
        win=win,
        name='dx', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.5, -0.25), size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "tutorial" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('tutorial.started', globalClock.getTime())
    dilemma_1_tutorial.setImage('images/dilemma_controllo_main_tutorial.jpg')
    audio_1_tutorial.setSound('audios/dilemma_controllo_main_tutorial.mp3', secs=14, hamming=True)
    audio_1_tutorial.setVolume(1.0, log=False)
    audio_1_tutorial.seek(0)
    dilemma_2_tutorial.setImage('images/dilemma_controllo_choice_tutorial.jpg')
    audio_2_tutorial.setSound('audios/dilemma_controllo_choice_tutorial.mp3', hamming=True)
    audio_2_tutorial.setVolume(1.0, log=False)
    audio_2_tutorial.seek(0)
    # setup some python lists for storing info about the mouse_tutorial
    mouse_tutorial.x = []
    mouse_tutorial.y = []
    mouse_tutorial.leftButton = []
    mouse_tutorial.midButton = []
    mouse_tutorial.rightButton = []
    mouse_tutorial.time = []
    mouse_tutorial.corr = []
    mouse_tutorial.clicked_name = []
    gotValidClick = False  # until a click is received
    sx_tutorial.setImage('images/sx.jpg')
    dx_tutorial.setImage('images/dx.jpg')
    # keep track of which components have finished
    tutorialComponents = [cross_tutorial, dilemma_1_tutorial, audio_1_tutorial, cross_2_tutorial, dilemma_2_tutorial, audio_2_tutorial, mouse_tutorial, sx_tutorial, dx_tutorial]
    for thisComponent in tutorialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "tutorial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_tutorial* updates
        
        # if cross_tutorial is starting this frame...
        if cross_tutorial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross_tutorial.frameNStart = frameN  # exact frame index
            cross_tutorial.tStart = t  # local t and not account for scr refresh
            cross_tutorial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_tutorial, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_tutorial.started')
            # update status
            cross_tutorial.status = STARTED
            cross_tutorial.setAutoDraw(True)
        
        # if cross_tutorial is active this frame...
        if cross_tutorial.status == STARTED:
            # update params
            pass
        
        # if cross_tutorial is stopping this frame...
        if cross_tutorial.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cross_tutorial.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                cross_tutorial.tStop = t  # not accounting for scr refresh
                cross_tutorial.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_tutorial.stopped')
                # update status
                cross_tutorial.status = FINISHED
                cross_tutorial.setAutoDraw(False)
        
        # *dilemma_1_tutorial* updates
        
        # if dilemma_1_tutorial is starting this frame...
        if dilemma_1_tutorial.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            dilemma_1_tutorial.frameNStart = frameN  # exact frame index
            dilemma_1_tutorial.tStart = t  # local t and not account for scr refresh
            dilemma_1_tutorial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dilemma_1_tutorial, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dilemma_1_tutorial.started')
            # update status
            dilemma_1_tutorial.status = STARTED
            dilemma_1_tutorial.setAutoDraw(True)
        
        # if dilemma_1_tutorial is active this frame...
        if dilemma_1_tutorial.status == STARTED:
            # update params
            pass
        
        # if dilemma_1_tutorial is stopping this frame...
        if dilemma_1_tutorial.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > dilemma_1_tutorial.tStartRefresh + 14-frameTolerance:
                # keep track of stop time/frame for later
                dilemma_1_tutorial.tStop = t  # not accounting for scr refresh
                dilemma_1_tutorial.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dilemma_1_tutorial.stopped')
                # update status
                dilemma_1_tutorial.status = FINISHED
                dilemma_1_tutorial.setAutoDraw(False)
        
        # if audio_1_tutorial is starting this frame...
        if audio_1_tutorial.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            audio_1_tutorial.frameNStart = frameN  # exact frame index
            audio_1_tutorial.tStart = t  # local t and not account for scr refresh
            audio_1_tutorial.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('audio_1_tutorial.started', tThisFlipGlobal)
            # update status
            audio_1_tutorial.status = STARTED
            audio_1_tutorial.play(when=win)  # sync with win flip
        
        # if audio_1_tutorial is stopping this frame...
        if audio_1_tutorial.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > audio_1_tutorial.tStartRefresh + 14-frameTolerance:
                # keep track of stop time/frame for later
                audio_1_tutorial.tStop = t  # not accounting for scr refresh
                audio_1_tutorial.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'audio_1_tutorial.stopped')
                # update status
                audio_1_tutorial.status = FINISHED
                audio_1_tutorial.stop()
        # update audio_1_tutorial status according to whether it's playing
        if audio_1_tutorial.isPlaying:
            audio_1_tutorial.status = STARTED
        elif audio_1_tutorial.isFinished:
            audio_1_tutorial.status = FINISHED
        
        # *cross_2_tutorial* updates
        
        # if cross_2_tutorial is starting this frame...
        if cross_2_tutorial.status == NOT_STARTED and tThisFlip >= 17-frameTolerance:
            # keep track of start time/frame for later
            cross_2_tutorial.frameNStart = frameN  # exact frame index
            cross_2_tutorial.tStart = t  # local t and not account for scr refresh
            cross_2_tutorial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_2_tutorial, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_2_tutorial.started')
            # update status
            cross_2_tutorial.status = STARTED
            cross_2_tutorial.setAutoDraw(True)
        
        # if cross_2_tutorial is active this frame...
        if cross_2_tutorial.status == STARTED:
            # update params
            pass
        
        # if cross_2_tutorial is stopping this frame...
        if cross_2_tutorial.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cross_2_tutorial.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                cross_2_tutorial.tStop = t  # not accounting for scr refresh
                cross_2_tutorial.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_2_tutorial.stopped')
                # update status
                cross_2_tutorial.status = FINISHED
                cross_2_tutorial.setAutoDraw(False)
        
        # *dilemma_2_tutorial* updates
        
        # if dilemma_2_tutorial is starting this frame...
        if dilemma_2_tutorial.status == NOT_STARTED and tThisFlip >= 20-frameTolerance:
            # keep track of start time/frame for later
            dilemma_2_tutorial.frameNStart = frameN  # exact frame index
            dilemma_2_tutorial.tStart = t  # local t and not account for scr refresh
            dilemma_2_tutorial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dilemma_2_tutorial, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dilemma_2_tutorial.started')
            # update status
            dilemma_2_tutorial.status = STARTED
            dilemma_2_tutorial.setAutoDraw(True)
        
        # if dilemma_2_tutorial is active this frame...
        if dilemma_2_tutorial.status == STARTED:
            # update params
            pass
        
        # if audio_2_tutorial is starting this frame...
        if audio_2_tutorial.status == NOT_STARTED and tThisFlip >= 20-frameTolerance:
            # keep track of start time/frame for later
            audio_2_tutorial.frameNStart = frameN  # exact frame index
            audio_2_tutorial.tStart = t  # local t and not account for scr refresh
            audio_2_tutorial.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('audio_2_tutorial.started', tThisFlipGlobal)
            # update status
            audio_2_tutorial.status = STARTED
            audio_2_tutorial.play(when=win)  # sync with win flip
        # update audio_2_tutorial status according to whether it's playing
        if audio_2_tutorial.isPlaying:
            audio_2_tutorial.status = STARTED
        elif audio_2_tutorial.isFinished:
            audio_2_tutorial.status = FINISHED
        # *mouse_tutorial* updates
        
        # if mouse_tutorial is starting this frame...
        if mouse_tutorial.status == NOT_STARTED and tThisFlip >= 20-frameTolerance:
            # keep track of start time/frame for later
            mouse_tutorial.frameNStart = frameN  # exact frame index
            mouse_tutorial.tStart = t  # local t and not account for scr refresh
            mouse_tutorial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_tutorial, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'mouse_tutorial.started')
            # update status
            mouse_tutorial.status = STARTED
            prevButtonState = mouse_tutorial.getPressed()  # if button is down already this ISN'T a new click
        if mouse_tutorial.status == STARTED:  # only update if started and not finished!
            buttons = mouse_tutorial.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames([sx_tutorial, dx_tutorial], namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_tutorial):
                            gotValidClick = True
                            mouse_tutorial.clicked_name.append(obj.name)
                    # check whether click was in correct object
                    if gotValidClick:
                        corr = 0
                        corrAns = environmenttools.getFromNames([], namespace=locals())
                        for obj in corrAns:
                            # is this object clicked on?
                            if obj.contains(mouse_tutorial):
                                corr = 1
                        mouse_tutorial.corr.append(corr)
                    if gotValidClick:
                        x, y = mouse_tutorial.getPos()
                        mouse_tutorial.x.append(x)
                        mouse_tutorial.y.append(y)
                        buttons = mouse_tutorial.getPressed()
                        mouse_tutorial.leftButton.append(buttons[0])
                        mouse_tutorial.midButton.append(buttons[1])
                        mouse_tutorial.rightButton.append(buttons[2])
                        mouse_tutorial.time.append(globalClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *sx_tutorial* updates
        
        # if sx_tutorial is starting this frame...
        if sx_tutorial.status == NOT_STARTED and tThisFlip >= 20-frameTolerance:
            # keep track of start time/frame for later
            sx_tutorial.frameNStart = frameN  # exact frame index
            sx_tutorial.tStart = t  # local t and not account for scr refresh
            sx_tutorial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sx_tutorial, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'sx_tutorial.started')
            # update status
            sx_tutorial.status = STARTED
            sx_tutorial.setAutoDraw(True)
        
        # if sx_tutorial is active this frame...
        if sx_tutorial.status == STARTED:
            # update params
            pass
        
        # *dx_tutorial* updates
        
        # if dx_tutorial is starting this frame...
        if dx_tutorial.status == NOT_STARTED and tThisFlip >= 20-frameTolerance:
            # keep track of start time/frame for later
            dx_tutorial.frameNStart = frameN  # exact frame index
            dx_tutorial.tStart = t  # local t and not account for scr refresh
            dx_tutorial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dx_tutorial, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dx_tutorial.started')
            # update status
            dx_tutorial.status = STARTED
            dx_tutorial.setAutoDraw(True)
        
        # if dx_tutorial is active this frame...
        if dx_tutorial.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in tutorialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "tutorial" ---
    for thisComponent in tutorialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('tutorial.stopped', globalClock.getTime())
    audio_1_tutorial.pause()  # ensure sound has stopped at end of Routine
    audio_2_tutorial.pause()  # ensure sound has stopped at end of Routine
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_tutorial.x', mouse_tutorial.x)
    thisExp.addData('mouse_tutorial.y', mouse_tutorial.y)
    thisExp.addData('mouse_tutorial.leftButton', mouse_tutorial.leftButton)
    thisExp.addData('mouse_tutorial.midButton', mouse_tutorial.midButton)
    thisExp.addData('mouse_tutorial.rightButton', mouse_tutorial.rightButton)
    thisExp.addData('mouse_tutorial.time', mouse_tutorial.time)
    thisExp.addData('mouse_tutorial.corr', mouse_tutorial.corr)
    thisExp.addData('mouse_tutorial.clicked_name', mouse_tutorial.clicked_name)
    thisExp.nextEntry()
    # the Routine "tutorial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('condition.xlsx'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "dilemmi" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('dilemmi.started', globalClock.getTime())
        dilemma_1.setImage(img_dilemma1)
        audio_1.setSound(audio_dilemma1, hamming=True)
        audio_1.setVolume(1.0, log=False)
        audio_1.seek(0)
        dilemma_2.setImage(img_dilemma2)
        audio_2.setSound(audio_dilemma2, hamming=True)
        audio_2.setVolume(1.0, log=False)
        audio_2.seek(0)
        # setup some python lists for storing info about the mouse
        mouse.x = []
        mouse.y = []
        mouse.leftButton = []
        mouse.midButton = []
        mouse.rightButton = []
        mouse.time = []
        mouse.corr = []
        mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        sx.setImage('images/sx.jpg')
        dx.setImage('images/dx.jpg')
        # keep track of which components have finished
        dilemmiComponents = [cross_img, dilemma_1, audio_1, cross_img_2, dilemma_2, audio_2, mouse, sx, dx]
        for thisComponent in dilemmiComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "dilemmi" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross_img* updates
            
            # if cross_img is starting this frame...
            if cross_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_img.frameNStart = frameN  # exact frame index
                cross_img.tStart = t  # local t and not account for scr refresh
                cross_img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_img.started')
                # update status
                cross_img.status = STARTED
                cross_img.setAutoDraw(True)
            
            # if cross_img is active this frame...
            if cross_img.status == STARTED:
                # update params
                pass
            
            # if cross_img is stopping this frame...
            if cross_img.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_img.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_img.tStop = t  # not accounting for scr refresh
                    cross_img.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_img.stopped')
                    # update status
                    cross_img.status = FINISHED
                    cross_img.setAutoDraw(False)
            
            # *dilemma_1* updates
            
            # if dilemma_1 is starting this frame...
            if dilemma_1.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                # keep track of start time/frame for later
                dilemma_1.frameNStart = frameN  # exact frame index
                dilemma_1.tStart = t  # local t and not account for scr refresh
                dilemma_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dilemma_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dilemma_1.started')
                # update status
                dilemma_1.status = STARTED
                dilemma_1.setAutoDraw(True)
            
            # if dilemma_1 is active this frame...
            if dilemma_1.status == STARTED:
                # update params
                pass
            
            # if dilemma_1 is stopping this frame...
            if dilemma_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dilemma_1.tStartRefresh + durata_audio1-frameTolerance:
                    # keep track of stop time/frame for later
                    dilemma_1.tStop = t  # not accounting for scr refresh
                    dilemma_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dilemma_1.stopped')
                    # update status
                    dilemma_1.status = FINISHED
                    dilemma_1.setAutoDraw(False)
            
            # if audio_1 is starting this frame...
            if audio_1.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                # keep track of start time/frame for later
                audio_1.frameNStart = frameN  # exact frame index
                audio_1.tStart = t  # local t and not account for scr refresh
                audio_1.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('audio_1.started', tThisFlipGlobal)
                # update status
                audio_1.status = STARTED
                audio_1.play(when=win)  # sync with win flip
            # update audio_1 status according to whether it's playing
            if audio_1.isPlaying:
                audio_1.status = STARTED
            elif audio_1.isFinished:
                audio_1.status = FINISHED
            
            # *cross_img_2* updates
            
            # if cross_img_2 is starting this frame...
            if cross_img_2.status == NOT_STARTED and tThisFlip >= start_cross2-frameTolerance:
                # keep track of start time/frame for later
                cross_img_2.frameNStart = frameN  # exact frame index
                cross_img_2.tStart = t  # local t and not account for scr refresh
                cross_img_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_img_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_img_2.started')
                # update status
                cross_img_2.status = STARTED
                cross_img_2.setAutoDraw(True)
            
            # if cross_img_2 is active this frame...
            if cross_img_2.status == STARTED:
                # update params
                pass
            
            # if cross_img_2 is stopping this frame...
            if cross_img_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_img_2.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_img_2.tStop = t  # not accounting for scr refresh
                    cross_img_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_img_2.stopped')
                    # update status
                    cross_img_2.status = FINISHED
                    cross_img_2.setAutoDraw(False)
            
            # *dilemma_2* updates
            
            # if dilemma_2 is starting this frame...
            if dilemma_2.status == NOT_STARTED and tThisFlip >= start_dilemma2-frameTolerance:
                # keep track of start time/frame for later
                dilemma_2.frameNStart = frameN  # exact frame index
                dilemma_2.tStart = t  # local t and not account for scr refresh
                dilemma_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dilemma_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dilemma_2.started')
                # update status
                dilemma_2.status = STARTED
                dilemma_2.setAutoDraw(True)
            
            # if dilemma_2 is active this frame...
            if dilemma_2.status == STARTED:
                # update params
                pass
            
            # if audio_2 is starting this frame...
            if audio_2.status == NOT_STARTED and tThisFlip >= start_dilemma2-frameTolerance:
                # keep track of start time/frame for later
                audio_2.frameNStart = frameN  # exact frame index
                audio_2.tStart = t  # local t and not account for scr refresh
                audio_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('audio_2.started', tThisFlipGlobal)
                # update status
                audio_2.status = STARTED
                audio_2.play(when=win)  # sync with win flip
            # update audio_2 status according to whether it's playing
            if audio_2.isPlaying:
                audio_2.status = STARTED
            elif audio_2.isFinished:
                audio_2.status = FINISHED
            # *mouse* updates
            
            # if mouse is starting this frame...
            if mouse.status == NOT_STARTED and tThisFlip >= start_dilemma2-frameTolerance:
                # keep track of start time/frame for later
                mouse.frameNStart = frameN  # exact frame index
                mouse.tStart = t  # local t and not account for scr refresh
                mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'mouse.started')
                # update status
                mouse.status = STARTED
                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
            if mouse.status == STARTED:  # only update if started and not finished!
                buttons = mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([sx, dx], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(mouse):
                                gotValidClick = True
                                mouse.clicked_name.append(obj.name)
                        # check whether click was in correct object
                        if gotValidClick:
                            corr = 0
                            corrAns = environmenttools.getFromNames([], namespace=locals())
                            for obj in corrAns:
                                # is this object clicked on?
                                if obj.contains(mouse):
                                    corr = 1
                            mouse.corr.append(corr)
                        if gotValidClick:
                            x, y = mouse.getPos()
                            mouse.x.append(x)
                            mouse.y.append(y)
                            buttons = mouse.getPressed()
                            mouse.leftButton.append(buttons[0])
                            mouse.midButton.append(buttons[1])
                            mouse.rightButton.append(buttons[2])
                            mouse.time.append(globalClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # *sx* updates
            
            # if sx is starting this frame...
            if sx.status == NOT_STARTED and tThisFlip >= start_dilemma2-frameTolerance:
                # keep track of start time/frame for later
                sx.frameNStart = frameN  # exact frame index
                sx.tStart = t  # local t and not account for scr refresh
                sx.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sx, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sx.started')
                # update status
                sx.status = STARTED
                sx.setAutoDraw(True)
            
            # if sx is active this frame...
            if sx.status == STARTED:
                # update params
                pass
            
            # *dx* updates
            
            # if dx is starting this frame...
            if dx.status == NOT_STARTED and tThisFlip >= start_dilemma2-frameTolerance:
                # keep track of start time/frame for later
                dx.frameNStart = frameN  # exact frame index
                dx.tStart = t  # local t and not account for scr refresh
                dx.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dx, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dx.started')
                # update status
                dx.status = STARTED
                dx.setAutoDraw(True)
            
            # if dx is active this frame...
            if dx.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in dilemmiComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "dilemmi" ---
        for thisComponent in dilemmiComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('dilemmi.stopped', globalClock.getTime())
        audio_1.pause()  # ensure sound has stopped at end of Routine
        audio_2.pause()  # ensure sound has stopped at end of Routine
        # store data for trials (TrialHandler)
        trials.addData('mouse.x', mouse.x)
        trials.addData('mouse.y', mouse.y)
        trials.addData('mouse.leftButton', mouse.leftButton)
        trials.addData('mouse.midButton', mouse.midButton)
        trials.addData('mouse.rightButton', mouse.rightButton)
        trials.addData('mouse.time', mouse.time)
        trials.addData('mouse.corr', mouse.corr)
        trials.addData('mouse.clicked_name', mouse.clicked_name)
        # the Routine "dilemmi" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials'
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    nest_asyncio.apply()
    device = discover_one_device()
    recording_id = device.recording_start()

    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
    device.recording_stop_and_save

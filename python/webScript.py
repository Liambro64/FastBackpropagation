import subprocess
import threading
from time import sleep
from traceback import print_list
from fastapi import background



isRunning = True
def readCommandFromFile(f : str):
	cmd = open("/mnt/c/Users/Cooke/Coding Projects/FastBackpropagation/python/cmd.txt").read(-1)
	fcmds = cmd.replace('\\\n', '\005').replace('  ', '').split("\005")
	startCommand = fcmds[0].split(' ')
	extras = fcmds[1:]
	fullCommands = startCommand[:2]
	for i in extras:
		fullCommands.append(i)
	return fullCommands
 
def print_list(lst):
	j = 0
	for i in lst:
		print("{0}: ".format(j)+i)
		j += 1
  
def readUntilRunFalse(stream):
    print("reading")
    global isRunning
    while isRunning:
        line = stream.stdout.readline()
        if not line:
            print("no line")
            break
        print(line.rstrip(), flush=True)
    print("Done reading")

def readLine(stream):
    line = stream.stdout.readline()
    return line

def waitForFinishOrSeconds(time_limit_s, sleep_time_ms, stream):
	print("Waiting")
	global isRunning
	sleep_time_s = sleep_time_ms / 1000
	time_current = 0
	while stream.poll() == None:
		sleep(sleep_time_s)
		time_current += sleep_time_s
		print(readLine(stream))
		if (time_current >= time_limit_s):
			print("Outta Time!")
			stream.kill()
			isRunning = False
			return
	stream.kill()
	isRunning = False
	print("Stream Ended Early")



print("Started")
filepath = "/mnt/c/Users/Cooke/Coding Projects/FastBackpropagation/python/cmd.txt"
command = ["bash", "-c", "'{0}'".format(filepath)]
proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

waitForFinishOrSeconds(5, 10, proc)


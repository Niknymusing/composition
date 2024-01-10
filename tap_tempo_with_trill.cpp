#include <Bela.h>
#include <cmath>
#include <libraries/Trill/Trill.h>
#include <libraries/Gui/Gui.h>
#include <libraries/OscSender/OscSender.h>
#include <libraries/OscReceiver/OscReceiver.h>
#include <mutex>

#define NUM_TOUCH 1 // Number of touches on Trill sensor

// Gui object declaration
Gui gui;

// Trill object declaration
Trill touchSensor;

// Location of touches on Trill Bar
float gTouchLocation[NUM_TOUCH] = { 0.0 };   // number of zeros must equal to NUM_TOUCH
// Size of touches on Trill bar
float gTouchSize[NUM_TOUCH] = { 0.0 };   // number of zeros must equal to NUM_TOUCH
// Number of active touches
int gNumActiveTouches = 0;

// Sleep time for auxiliary task
unsigned int gTaskSleepTime = 12000; // microseconds
// Time period (in seconds) after which data will be sent to the GUI
float gTimePeriod = 0.015;

OscReceiver oscReceiver;
OscSender oscSender;

int localPort = 7562;
int remotePort = 7564;
const char* remoteIp = "192.168.7.1"; //"192.168.1.120";

// Add a global variable to count the number of recorded audio samples
unsigned long long gAudioSamples = 0;  // Total number of audio samples processed
double gElapsedAudioTime = 0.0;  // Total elapsed audio time
double gOscMessageDelay = 0.1;  // Delay time for sending OSC messages (in seconds)

// Interval struct to hold intervals
struct Interval {
    unsigned long long start;
    unsigned long long end;
};
std::vector<Interval> gIntervals;

std::mutex gIntervalsMutex;

void on_receive(oscpkt::Message* msg, const char* addr, void* arg)
{
    if(msg->match("/interval")) {
        oscpkt::Message::ArgReader arg = msg->arg();
        while(arg.isOk()) {
            int32_t start, end;
            Interval newInterval;
            if(arg.popInt32(start) && arg.popInt32(end)) {
                newInterval.start = start;
                newInterval.end = end;
                gIntervalsMutex.lock();
                gIntervals.push_back(newInterval);
                gIntervalsMutex.unlock();
                //rt_printf("Received interval: start = %d, end = %d\n", start, end);  // Debug print
            }
        }
    } else if(msg->match("/checktime")) {
        gIntervalsMutex.lock();
        std::string gAudioSamplesStr = std::to_string(gAudioSamples);
        gIntervalsMutex.unlock();
        oscSender.newMessage("/time").add(gAudioSamplesStr).send();
        //rt_printf("Received checktime, sent audio samples: %s\n", gAudioSamplesStr.c_str());  // Debug print
    }
}



// Global variables
//unsigned int gLastOscMessageSample[NUM_TOUCH] = { 0 }; // number of zeros must equal to NUM_TOUCH
//unsigned int gOscMessageDelaySamples; // This will be calculated in setup()
unsigned int gSampleRate; // This will be set in setup()
// Global variable to keep track of the last OSC message sent time
double gLastOscMessageTime = 0.0;

void loop(void*)
{
	while(!Bela_stopRequested())
	{
		// Read locations from Trill sensor
		touchSensor.readI2C();
		gNumActiveTouches = touchSensor.getNumTouches();
		// Inside the loop() function
		for(unsigned int i = 0; i < gNumActiveTouches; i++) {
		    // Read touch information
		    gTouchLocation[i] = touchSensor.touchLocation(i);
		    gTouchSize[i] = touchSensor.touchSize(i);
		
		    // Calculate the current time
		    double currentTime = gAudioSamples / (double)gSampleRate;
		
		    // If enough time has passed since the last OSC message
		    if(currentTime - gLastOscMessageTime >= gOscMessageDelay)
		    {
		        // Convert the number of samples to a string
		        std::string audioSamplesStr = std::to_string(gAudioSamples);
		
		        // Send OSC message with only the timestamp
		        oscSender.newMessage("/trill").add(audioSamplesStr).send();
		        // Update the time when the last OSC message was sent
		        gLastOscMessageTime = currentTime;
		    }
		}
	

		// For all inactive touches, set location and size to 0
		for(unsigned int i = gNumActiveTouches; i < NUM_TOUCH; i++) {
			gTouchLocation[i] = 0.0;
			gTouchSize[i] = 0.0;
			// Also reset the last OSC message sample count for inactive touch points
			//gLastOscMessageSample[i] = 0;
		}
		usleep(gTaskSleepTime);
	}
}


bool setup(BelaContext *context, void *userData)
{
	if(touchSensor.setup(1, Trill::BAR) != 0) {
		fprintf(stderr, "Unable to initialise Trill Bar\n");
		return false;
	}

	touchSensor.printDetails();

	Bela_runAuxiliaryTask(loop);

	gui.setup(context->projectName);

	oscReceiver.setup(localPort, on_receive);
	oscSender.setup(remotePort, remoteIp);

	gSampleRate = context->audioSampleRate;
	//gOscMessageDelaySamples = gSampleRate * gOscMessageDelay;
	
	return true;
}

void render(BelaContext *context, void *userData)
{
    for(unsigned int n = 0; n < context->audioFrames; n++) {
        gAudioSamples++;
        gElapsedAudioTime = (double)gAudioSamples / context->audioSampleRate;

        for(unsigned int channel = 0; channel < context->audioOutChannels; channel++) {
            float in = audioRead(context, n, channel);
            audioWrite(context, n, channel, in);
        }

        gIntervalsMutex.lock();
        for (auto it = gIntervals.begin(); it != gIntervals.end();) {
            if (gAudioSamples >= it->start && gAudioSamples <= it->end) {
                for(unsigned int channel = 0; channel < context->audioOutChannels; channel++) {
                    float in = audioRead(context, n, channel + context->audioInChannels / 2);  // assuming that second half of audioInChannels are used for the second audio input
                    audioWrite(context, n, channel, in);
                }
            }
            if (gAudioSamples > it->end) {
                it = gIntervals.erase(it);
            } else {
                ++it;
            }
        }
        gIntervalsMutex.unlock();
    }
}

void cleanup(BelaContext *context, void *userData)
{}

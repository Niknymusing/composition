#include <Bela.h>
#include <libraries/OscSender/OscSender.h>
#include <libraries/OscReceiver/OscReceiver.h>
#include <chrono>
#include <deque>
#include <string>
#include <vector>

// Osc setup
int localPort = 12345;  // Port to receive messages
int remotePort = 8000; // Port to send messages
std::string remoteIp = "192.168.7.1"; // IP to send messages

OscSender sender;
OscReceiver receiver;
unsigned long long sampleCounter = 0; // Counts the samples processed

// Structure for queued messages
struct ScheduledMessage {
    unsigned long long scheduleTime;  // When to send the message
    std::vector<float> responseArgs;  // Vector to hold the float arguments for the response
};

// Queue to hold scheduled messages
std::deque<ScheduledMessage> messageQueue;

// Callback function for handling incoming OSC messages
void on_receive(oscpkt::Message* msg, const char* addr, void* arg){
    if(msg) {
        int samplesToWait;
        std::vector<float> responseArgs;  // Vector to hold float arguments

        oscpkt::Message::ArgReader argReader = msg->arg();
        argReader.popInt32(samplesToWait);

        // Then, try to pop an array of numbers (integers or floats)
        while(argReader.nbArgRemaining() > 0) {
            if (argReader.isInt32()) {
                int value;
                argReader.popInt32(value);
                responseArgs.push_back(static_cast<float>(value));
            } else if (argReader.isFloat()) {
                float value;
                argReader.popFloat(value);
                responseArgs.push_back(value);
            } else {
                break;  // Break if neither int nor float
            }
        }

        if (argReader.isOk()) {
            // Queue the new message with its scheduled send time and response content
            messageQueue.push_back({sampleCounter + samplesToWait, responseArgs});
            rt_printf("OSC message received and queued at sample count: %llu, to be sent after %d samples\n", sampleCounter, samplesToWait);
        } else {
            rt_printf("Error reading arguments from OSC message\n");
        }
    }
}

bool setup(BelaContext *context, void *userData)
{
    sender.setup(remotePort, remoteIp); // Setup sender with remote port and IP
    receiver.setup(localPort, &on_receive, nullptr); // Setup receiver with local port and callback function

    return true;
}


void render(BelaContext *context, void *userData)
{
    // Increment the sample counter for each audio frame processed
    for(unsigned int n = 0; n < context->audioFrames; n++) {
        sampleCounter++;

        // Check the queue for any messages ready to be sent
        while(!messageQueue.empty() && sampleCounter >= messageQueue.front().scheduleTime) {
            // Construct and send the OSC message with the queued content
            oscpkt::Message newMsg;
            newMsg.init("/response");  // Initialize the new OSC message
            for (float arg : messageQueue.front().responseArgs) {
                newMsg.pushFloat(arg);
            }
            sender.send(newMsg);  // Send the constructed message
            rt_printf("OSC message sent at sample count: %llu\n", sampleCounter);

            // Remove the message from the queue
            messageQueue.pop_front();
        }
    }
}


void cleanup(BelaContext *context, void *userData)
{
    // Any cleanup code goes here
}


int main(int argc, char **argv)
{
    BelaInitSettings settings; // Standard setup for Bela programs
    Bela_defaultSettings(&settings);
    settings.setup = setup; // Your setup function
    settings.render = render; // Your render function
    settings.cleanup = cleanup; // Your cleanup function

    // Initialize the Bela application
    if(Bela_initAudio(&settings, 0) != 0) {
        fprintf(stderr, "Error initializing audio\n");
        return 1;
    }

    // Run the Bela application
    Bela_startAudio();

    // This is typically an infinite loop that keeps your application running
    while (!gShouldStop) {
        // Your application's real-time control loop could go here
        usleep(10000); // Sleeping is generally fine in the main loop (not in render)
    }

    // Clean up the Bela application
    Bela_stopAudio();
    Bela_cleanupAudio();

    return 0;
}

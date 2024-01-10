#include <Bela.h>
#include <libraries/OscSender/OscSender.h>
#include <libraries/OscReceiver/OscReceiver.h>
#include <deque>
#include <numeric>

// Osc setup
int localPort = 12345;  // Port to receive messages
int remotePort = 8008; // Port to send messages
std::string remoteIp = "192.168.7.1"; // IP to send messages

OscSender sender;
OscReceiver receiver;
unsigned long long sampleCounter = 0; // Counts the samples processed

// Define the ScheduledMessage structure
struct ScheduledMessage {
    unsigned long long scheduleTime;  // When to send the message
    std::vector<float> responseArgs;  // Vector to hold the float arguments for the response
};

std::deque<unsigned long long> tapTimes;
const unsigned long long maxSampleDifference = 100000; // Maximum sample difference between taps
int tempo = 0;

std::deque<ScheduledMessage> messageQueue;

void calculateTempo() {
    while(tapTimes.size() > 1 && (tapTimes.back() - tapTimes.front()) > maxSampleDifference) {
        tapTimes.pop_front(); // Remove oldest tap if it's too old
    }

    if(tapTimes.size() > 1) {
        std::vector<unsigned long long> intervals;
        for(size_t i = 1; i < tapTimes.size(); ++i) {
            intervals.push_back(tapTimes[i] - tapTimes[i-1]);
        }

        unsigned long long sum = std::accumulate(intervals.begin(), intervals.end(), 0ULL);
        tempo = static_cast<int>(sum / intervals.size());
    }
}

// Callback function for handling incoming OSC messages
void on_receive(oscpkt::Message* msg, const char* addr, void* arg){
    if(msg) {
        if(msg->match("/trill")) {
            oscpkt::Message::ArgReader argReader = msg->arg();
            int64_t timestamp;
            if(argReader.popInt64(timestamp).isOkNoMoreArgs()) {
                tapTimes.push_back(static_cast<unsigned long long>(timestamp));
                calculateTempo();
            }
        } else {
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
    BelaInitSettings settings;
    Bela_defaultSettings(&settings);
    settings.setup = setup;
    settings.render = render;
    settings.cleanup = cleanup;

    if(Bela_initAudio(&settings, 0) != 0) {
        fprintf(stderr, "Error initializing audio\n");
        return 1;
    }

    Bela_startAudio();

    while (!gShouldStop) {
        usleep(10000);
    }

    Bela_stopAudio();
    Bela_cleanupAudio();

    return 0;
}

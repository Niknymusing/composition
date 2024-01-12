#include <Bela.h>
#include <libraries/OscSender/OscSender.h>
#include <libraries/OscReceiver/OscReceiver.h>
#include <deque>
#include <numeric>
#include <sstream> 
#include <mutex>



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
int tempo = 10000;

std::deque<ScheduledMessage> messageQueue;
std::mutex tempoMutex;





void calculateTempo() {
    // Ensure at least 2 samples for calculation
    if(tapTimes.size() >= 2) {
        // Get the last two tap times
        unsigned long long lastTap = tapTimes.back();
        unsigned long long secondLastTap = tapTimes[tapTimes.size() - 2];

        // Calculate the interval and check against maxSampleDifference
        unsigned long long interval = lastTap - secondLastTap;
        if(interval > 0 && interval <= maxSampleDifference) {
            tempo = static_cast<int>(interval);
            double bpm = (44100.0 / tempo) * 60.0; // Convert to BPM
            rt_printf("Updated tempo: %f BPM\n", bpm);
        } else {
            rt_printf("Interval is invalid (zero, too large, or more than maxSampleDifference), cannot update tempo\n");
        }

        // Remove all but the last tap time to ensure only the last two are used next time
        while(tapTimes.size() > 1) {
            tapTimes.pop_front();
        }
    } else {
        rt_printf("Not enough tap times to calculate tempo\n");
    }
}






void parseAndScheduleMessage(const std::string& rhythmInfo) {
    // Parse the rhythm string
    //std::lock_guard<std::mutex> lock(tempoMutex);
    std::istringstream iss(rhythmInfo);
    std::vector<int> values(6);
    char comma;
    for (int i = 0; i < 6; ++i) {
        if (!(iss >> values[i]) || (i < 5 && !(iss >> comma && comma == ','))) {
            rt_printf("Error parsing rhythm information\n");
            return;
        }
    }

    // Calculate the time for the next event
    unsigned long long next_event_time_incr = 0;
    for (int i = 0; i < values.size(); ++i) {
        next_event_time_incr += tempo * values[i] / (1ULL << i);
    }

    // Schedule the message
    unsigned long long scheduleTime = sampleCounter + next_event_time_incr;
    std::vector<float> responseArgs; // This vector should be filled as needed
    messageQueue.push_back({scheduleTime, responseArgs});
}


unsigned long long calculateNextEventTime(const std::vector<int>& rhythmValues) {
    unsigned long long next_event_time_incr = 0;
    for (int i = 0; i < rhythmValues.size(); ++i) {
        next_event_time_incr += tempo * rhythmValues[i] / (1 << i);
    }
    return next_event_time_incr;
}


// Callback function for handling incoming OSC messages
void on_receive(oscpkt::Message* msg, const char* addr, void* arg){
    if(msg) {
        if(msg->match("/tempo")) {
            // Push the current sampleCounter instead of the timestamp
            tapTimes.push_back(sampleCounter);
            calculateTempo();
        	
        } else if (msg->match("/rhythm")) {
            // Handle rhythm information
            std::string rhythmInfo;
            if (msg->arg().popStr(rhythmInfo).isOkNoMoreArgs()) {
                // Parse the rhythm string
                std::istringstream iss(rhythmInfo);
                std::vector<int> values(6);
                char comma;
                for (int i = 0; i < 6; ++i) {
                    if (!(iss >> values[i]) || (i < 5 && !(iss >> comma && comma == ','))) {
                        rt_printf("Error parsing rhythm information\n");
                        return;
                    }
                }

                // Calculate the time for the next event
                unsigned long long next_event_time_incr = 0;
                for (int i = 0; i < values.size(); ++i) {
                    next_event_time_incr += tempo * values[i] / (1 << i);
                }

                // Calculate schedule time based on the last message sent
                unsigned long long lastMessageTime = messageQueue.empty() ? sampleCounter : messageQueue.back().scheduleTime;
                unsigned long long scheduleTime = lastMessageTime + next_event_time_incr;
                
                std::vector<float> responseArgs; // This should be filled as needed
                messageQueue.push_back({scheduleTime, responseArgs});
            } else {
                rt_printf("Error reading rhythm information from OSC message\n");
            }    
        
        } else {
            // General OSC message handling
            // [Rest of the code remains unchanged]
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

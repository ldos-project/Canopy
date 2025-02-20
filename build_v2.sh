pushd ~/ConstrainedOrca
    g++ -pthread src/orca-server-mahimahi_v2.cc src/flow.cc -o orca-server-mahimahi_v2
    g++ -pthread src/tcp_eval.cc src/flow.cc -o tcp_eval
    g++ -pthread src/real_world.cc src/flow.cc -o real_world
    g++ src/client.c -o client
    cp client rl-module/
    mv orca-server*  rl-module/
    mv tcp_eval rl-module/
    mv real_world rl-module
    sudo chmod +x rl-module/client
    sudo chmod +x rl-module/orca-server-mahimahi_v2
    sudo chmod +x rl-module/tcp_eval
    sudo chmod +x rl-module/real_world
popd
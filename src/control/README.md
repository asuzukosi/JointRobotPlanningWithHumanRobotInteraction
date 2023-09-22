## In this section we will implement low-level primitives for robot operation

We combine the models used in vision and language to generate robot behaviour for the robot to perform
The logic behind the control module is to receive a text prompt from the user and pass it into a language model
The language model generates a code fuction which uses pre-defined primitives as APIs and performs the intended logic.
Althouhgh we don't have a system for ensuring the generated code is correct we can implement a system that checks for the status code when thie command to start the action node is called. 
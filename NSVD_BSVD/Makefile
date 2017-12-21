export CC = gcc
export CXX = g++
export RM = rm
export CFLAGS = -Wall -Werror -O3 -msse2
export CXXFLAGS = -Wall -O3 -msse2
export LDFLAGS = -lpthread -lm

EXEC = demo
OBJ = Predictor_data.o Predictor_train.o Predictor_test.o SGD.o SVDBiasPredictor_init.o  SVDBiasPredictor_update.o SVDBiasPredictor_data.o

.PHONY: clean all

all : $(OBJ) $(EXEC)


demo : CF_main.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^


#%.o : %.c
#	$(CXX) -c $(CXXFLAGS) $< -o $@


clean:
	$(RM) $(OBJ) $(EXEC) 


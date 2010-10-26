
all: definitions.h pt_test_3d.cpp shared.cpp shared.h timer.cpp timer.h test_scheduler
#	g++ -O3 -lpthread pt_test_3d.cpp shared.cpp timer.cpp -msse2 -DNAIVE -D_4D -DLB

test_scheduler: test_schedule.cpp scheduler.h
	g++ -g -o test_scheduler test_schedule.cpp
clean:
	rm -f a.out test_scheduler
BUILD_DIR ?= ./build

CPPFLAGS ?= -MMD -MP -Wall -std=c++17 -O3 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vbmi2

CXX=g++

.PHONY: all test perf clean
all: test perf
test: $(BUILD_DIR)/test
perf: $(BUILD_DIR)/perf
clean:
	$(RM) -r $(BUILD_DIR)

$(BUILD_DIR)/test: $(BUILD_DIR)/test.cpp.o
	$(CXX) $< -o $@

ifeq ($(origin IPPROOT), undefined)
$(BUILD_DIR)/perf: $(BUILD_DIR)/perf.cpp.o
	$(CXX) $< -o $@
else
$(BUILD_DIR)/perf: $(BUILD_DIR)/perf.cpp.o
	$(CXX) -L$(IPPROOT)/lib/intel64 -lippcore -lipps $< -o $@
endif


$(BUILD_DIR)/test.cpp.o: test.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) -c $< -o $@

ifeq ($(origin IPPROOT), undefined)
$(BUILD_DIR)/perf.cpp.o: perf.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) -c $< -o $@
else
$(BUILD_DIR)/perf.cpp.o: perf.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) -I$(IPPROOT)/include -D_IPP_RADIX_IS_PRESENT_ $(CPPFLAGS) -c $< -o $@
endif


-include $(BUILD_DIR)/test.cpp.d $(BUILD_DIR)/perf.cpp.d

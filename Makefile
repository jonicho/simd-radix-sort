BUILD_DIR ?= ./build

CPPFLAGS ?= -MMD -MP -Wall -std=c++20 -O3 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vbmi2

CXX=g++

BINARIES = $(patsubst %.cpp,%,$(wildcard *.cpp))

.PHONY: all clean $(BINARIES)
all: $(BINARIES)
$(BINARIES): %: $(BUILD_DIR)/%
clean:
	$(RM) -r $(BUILD_DIR)

ifeq ($(origin IPPROOT), undefined)
IPPFLAGS =
else
IPPFLAGS = -I$(IPPROOT)/include -D_IPP_RADIX_IS_PRESENT_ -L$(IPPROOT)/lib/intel64 -lippcore -lipps
endif

$(addprefix $(BUILD_DIR)/,$(BINARIES)): $(BUILD_DIR)/%: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(IPPFLAGS) $< -o $@

-include $(BUILD_DIR)/*.d

CXX      := mpiCC
CXXFLAGS := -std=gnu++11 -Wall 
BRAID_DIR:= libs/xbraid/braid/
LDFLAGS  := -L$(BRAID_DIR) -lbraid -ldl -lstdc++fs 
BUILD    := ./build
OBJ_DIR  := $(BUILD)/objects
APP_DIR  := $(BUILD)/apps
TARGET   := program
INCLUDE  := -Iinclude/ -I$(BRAID_DIR)
SRC      := $(wildcard src/*.cpp)

OBJECTS  := $(SRC:%.cpp=$(OBJ_DIR)/%.o)

ModelsDir := $(shell find  models/ -maxdepth 1 -mindepth 1 -type d)

all: build $(APP_DIR)/$(TARGET) $(ModelsDir)

mgrit: build $(APP_DIR)/$(TARGET)


$(ModelsDir): 
	cmake -S $@ -B $@ 
	@$(MAKE) --no-print-directory -C $@ all
	
fixed:
	cmake -D CMAKE_CXX_FLAGS="-DFIXED" -S models/TorchXorPaper -B models/TorchXorPaper
	@$(MAKE) --no-print-directory -C models/TorchXorPaper all

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@ 

$(APP_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)

.PHONY: all build clean debug release $(ModelsDir)

build:
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all

profile: CXXFLAGS += -pg 
profile: LDFLAGS += -pg
profile: all

release: CXXFLAGS += -O2
release: all

clean: $(ModelsDir:=clean)
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf $(APP_DIR)/*

$(ModelsDir:=clean):
	@$(MAKE) --no-print-directory -C $(patsubst %clean,%,$@) clean


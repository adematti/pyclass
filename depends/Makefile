CLASS_VERSION?=3.2.0
DEST?=_inst

TARBALL=class-v$(CLASS_VERSION).tar.gz
UNPACK=tmp-class-v$(CLASS_VERSION)
SRC=$(UNPACK)/class_public-$(CLASS_VERSION)

$(TARBALL):
	curl -fsSL https://github.com/adematti/class_public/archive/refs/tags/v$(CLASS_VERSION).tar.gz > $(TARBALL)
	#curl -fsSL https://github.com/lesgourg/class_public/archive/refs/tags/v$(CLASS_VERSION).tar.gz > $(TARBALL)
	#wget -O $(TARBALL) https://github.com/adematti/class_public/archive/refs/tags/v$(CLASS_VERSION).tar.gz

$(SRC): $(TARBALL)
	mkdir -p $(UNPACK)
	gzip -dc $(TARBALL) | tar xf - -C $(UNPACK)
	touch $@

$(SRC)/libclass.a: $(SRC)
	cd $(SRC); make libclass.a

$(DEST)/lib/libclass.a: $(SRC)/libclass.a
	mkdir -p $(DEST)
	cp -r $(SRC)/include $(DEST)
	cp -r $(SRC)/external $(DEST)
	mkdir -p $(DEST)/lib
	cp $(SRC)/libclass.a $(DEST)/lib
	mkdir -p $(DEST)/data
	cp $(SRC)/*.ini $(SRC)/*.pre $(DEST)/data/
	rm -rf $(TARBALL) $(UNPACK) $(SRC)

build: $(SRC)/libclass.a
install: $(DEST)/lib/libclass.a
clean:
	rm -rf $(DEST)

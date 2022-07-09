LOCAL_IP = '$(shell ifconfig |grep "12.12.12.1"| grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | awk '{print $1}')'

all:
	@echo "What you want to do?";
	@echo "make sync: 同步代码";
	@echo "make copy: 复制代码";

sync:
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.101:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.102:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.103:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.104:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.105:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.106:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.107:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.108:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.109:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.110:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.111:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.112:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.113:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.114:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.115:~/sync_test/allreduce_over_mpi
	scp ./build/allreduce_over_mpi dictxiong@12.12.12.116:~/sync_test/allreduce_over_mpi

build:clean
	mkdir -p build; cd build; cmake ..; make -j32;

.PHONY:clean
clean:
	rm -rf ./build;
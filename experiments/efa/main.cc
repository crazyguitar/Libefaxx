#include <rdma/fabric/efa.h>
#include <rdma/ib/efa.h>
#include <iostream>

int main(int argc, char *argv[]) {
  std::cout << "=== libfabric EFA devices ===\n";
  auto* info = EFAInfo::Get();
  for (auto p = info; !!p; p = p->next) {
    std::cout << "provider: " << p->fabric_attr->prov_name << "\n";
    std::cout << "    fabric: " << p->fabric_attr->name << "\n";
    std::cout << "    domain: " << p->domain_attr->name << "\n";
  }

  std::cout << "\n=== ibverbs EFA devices ===\n";
  auto& devs = VerbDeviceList::Get();
  for (auto* dev : devs.EFA()) {
    VerbDevice efa(dev);
    std::cout << "provider: efa\n";
    std::cout << "    fabric: efa\n";
    std::cout << "    domain: " << efa.Name() << "\n";
  }
}

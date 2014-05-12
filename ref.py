import ipaddress, sys, random, time
if len(sys.argv) < 2:
    print("Usage: %s n" % sys.argv[0], file=sys.stderr)
    sys.exit(1)

N = int(sys.argv[1])
b = ipaddress._BaseV4("")

ips = list()
for i in range(0, N):
    ips.append(str(ipaddress.ip_address(random.randint(0, 0xFFFFFFFF))))

start = time.time()
for ip in ips:
    b._ip_int_from_string(ip)
end = time.time()
diff = end-start
print("Done in %0.4f s => %0.4f conversions/s" % (diff, N/diff))

#!/usr/bin/env python3

# ryzen-power: measure AMD Ryzen CPU power consumption.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
# This program is a Python port of rapl-read-ryzen
# https://github.com/djselbeck/rapl-read-ryzen

import logging
import os.path
import argparse
from itertools import count
from struct import unpack
from time import sleep
from warnings import warn

logger = logging.getLogger("ryzen-master")


class RyzenPower:
    AMD_MSR_PWR_UNIT_OFFSET = 0xC0010299
    AMD_MSR_CORE_ENERGY_OFFSET = 0xC001029A
    AMD_MSR_PACKAGE_ENERGY_OFFSET = 0xC001029B
    AMD_TIME_UNIT_MASK = 0xF0000
    AMD_ENERGY_UNIT_MASK = 0x1F00
    AMD_POWER_UNIT_MASK = 0xF

    def __init__(self, duration=1.0):
        self._energy_unit = self._get_energy_units()
        self._is_smt = self._detect_smt()
        self._package_topology = self._detect_physical_package_topology()
        self._duration = duration
        self._cores = list(self._package_topology.keys())
        if self._is_smt:
            self._cores = [c for c in self._cores if c % 2 == 0]
        self._cores = sorted(self._cores)
        self._msr_fd_cache = {}

    @staticmethod
    def _read(filename):
        with open(filename, "r") as f:
            return f.read()

    def _detect_smt(self):
        try:
            smt_status = self._read("/sys/devices/system/cpu/smt/control").strip()
            logger.debug("CPU smt status is {}".format(smt_status))
            return smt_status == "on"
        except FileNotFoundError:
            warn("unable to detect CPU SMT status, assume SMT is on")
            return True

    @staticmethod
    def _detect_physical_package_topology():
        cpu_package_mapping = {}
        for cpu_id in count():
            filename = "/sys/devices/system/cpu/cpu{}/topology/physical_package_id".format(cpu_id)
            if os.path.isfile(filename):
                with open(filename, "r") as f:
                    package_id = int(f.read())
                logger.debug("detected cpu {} in socket {}".format(cpu_id, package_id))
                cpu_package_mapping[cpu_id] = package_id
            else:
                return cpu_package_mapping

    def _read_msr(self, cpu_id, offset):
        msr_file = "/dev/cpu/{}/msr".format(cpu_id)
        try:
            with open(msr_file, "rb") as f:
                f.seek(offset)
                # MSR value is always 64 bits
                # https://manpages.debian.org/buster/manpages/msr.4.en.html
                return self._decode_int64(f.read(8))
        except PermissionError:
            raise PermissionError("root privilege is required to read model-specific registers")
        except FileNotFoundError:
            raise FileNotFoundError("msr driver is not loaded, try \"sudo modprobe msr\" to load msr module")

    @staticmethod
    def _decode_int64(buffer):
        return unpack("q", buffer)[0]

    def _read_all_units(self):
        return self._read_msr(0, self.AMD_MSR_PWR_UNIT_OFFSET)

    def _get_energy_units(self):
        energy_unit = (self._read_all_units() & self.AMD_ENERGY_UNIT_MASK) >> 8
        logger.debug("CPU energy unit is 1/2^{}".format(energy_unit))
        energy_unit = 0.5 ** energy_unit
        return energy_unit

    def _read_package_energy(self, cpu_id):
        energy = self._read_msr(cpu_id, self.AMD_MSR_PACKAGE_ENERGY_OFFSET)
        logger.debug("CPU {} current package energy {} J".format(cpu_id, energy, self._energy_unit))
        return energy

    def _read_core_energy(self, cpu_id):
        energy = self._read_msr(cpu_id, self.AMD_MSR_CORE_ENERGY_OFFSET)
        logger.debug("CPU {} current core energy {} * {} J".format(cpu_id, energy, self._energy_unit))
        return energy

    def _calc_power(self, before, after):
        return (after - before) * self._energy_unit / self._duration

    def measure(self):
        package_energy_before = {c: self._read_package_energy(c) for c in self._cores}
        core_energy_before = {c: self._read_core_energy(c) for c in self._cores}
        logger.debug("sleep for {} seconds".format(self._duration))
        sleep(self._duration)
        package_energy_after = {c: self._read_package_energy(c) for c in self._cores}
        core_energy_after = {c: self._read_core_energy(c) for c in self._cores}
        package_power = {c: self._calc_power(package_energy_before[c], package_energy_after[c]) for c in self._cores}
        core_power = {c: self._calc_power(core_energy_before[c], core_energy_after[c]) for c in self._cores}
        print(self._format_result(package_power, core_power))

    @staticmethod
    def _format_table(table, widths, units):
        buffer = []
        for row in table:
            row_buffer = []
            for col, width, unit in zip(row, widths, units):
                if isinstance(col, float):
                    row_buffer.append("{:.2f}{}".format(col, unit).ljust(width))
                else:
                    row_buffer.append(str(col).ljust(width))
            buffer.append("".join(row_buffer))
        return "\n".join(buffer)

    def _format_result(self, package_power, core_power):
        sockets = sorted(set(self._package_topology.values()))
        table = [["", "Cores Power", "Package Power"]]
        for socket in sockets:
            socket_total_cores_power = 0
            socket_package_power = 0
            socket_power_entry = ["SOCKET {: 2}:".format(socket)]
            table.append(socket_power_entry)
            for core in self._cores:
                if self._package_topology[core] == socket:
                    socket_total_cores_power += core_power[core]
                    socket_package_power = package_power[core]
                    table.append([
                        "  CORE {: 2}:".format(core // 2 if self._is_smt else core),
                        core_power[core],
                        ""
                    ])
            socket_power_entry.append(socket_total_cores_power)
            socket_power_entry.append(socket_package_power)
        return self._format_table(table, (16, 16, 16), ("", "W", "W"))


parser = argparse.ArgumentParser(description='Measure power consumption for AMD Ryzen CPU')
parser.add_argument("--debug", action='store_true', help="show debug messages")
parser.add_argument("-d", "--duration", type=float, default=0.5,
                    help="the duration of measurement in seconds, default is 0.5 second")
args = parser.parse_args()
if args.debug:
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)

RyzenPower(args.duration).measure()

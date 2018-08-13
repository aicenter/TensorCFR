#!/usr/bin/env python3
import datetime


def get_current_timestamp():
	return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

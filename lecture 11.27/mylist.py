# -*- coding: utf-8 -*-
"""
JasonGUTU
OOP Demo"""


class Node(object):
    """Node in linked list"""
    def __init__(self, value, next_node=None):
        self.__value = value
        if next_node is not None:
            assert isinstance(next_node, Node), "`next_node` must be type Node."
            self.__next = next_node
        else:
            self.__next = None

    def __str__(self):
        msg = "<Node with value:%s>" % self.__value
        return msg

    def value(self):
        return self.__value

    def change(self, value):
        self.__value = value

    def redirect(self, next_node):
        assert isinstance(next_node, Node), "redirect must be a Node."
        self.__next = next_node

    def next(self):
        return self.__next


class LinkedList(object):
    """Linked list use node"""
    def __init__(self):
        self.__head = Node(None)
        self.__len = 0

    def add_head(self, value):
        pre_node = self.__head.next()
        node = Node(value, next_node=pre_node)
        self.__head.redirect(node)
        self.__len += 1

    def add_tail(self, value):
        node = Node(value)
        tail_node = self.__head
        for i in range(self.__len):
            tail_node = tail_node.next()
        tail_node.redirect(node)
        self.__len += 1

    def insert(self, position, value):
        assert isinstance(position, int), "`posiotion` must be int."
        assert position > 0 and position <= self.__len, "index out of range."
        pre_node = self.__head
        for i in range(position):
            pre_node = pre_node.next()
        nxt_node = pre_node.next()
        insert_node = Node(value, nxt_node)
        pre_node.redirect(insert_node)
        self.__len += 1

    def delete(self, position):
        assert isinstance(position, int), "`posiotion` must be int."
        assert position > 0 and position <= self.__len, "index out of range."
        pre_node = self.__head
        for i in range(position - 1):
            pre_node = pre_node.next()
        del_node = pre_node.next()
        nxt_node = del_node.next()
        pre_node.redirect(nxt_node)
        self.__len -= 1

    def __len__(self):
        return self.__len

    def __str__(self):
        msg = ''
        node = self.__head
        for i in range(self.__len):
            node = node.next()
            msg += "%d : %s\n" % (i + 1, node.value())
        return msg

    def _objectOf(self, index):
        assert isinstance(index, int), "`index` must be int type."
        assert index <= self.__len and index >0, "index out of range."
        node = self.__head 
        for i in range(index):
            node = node.next()
        return node

    def _len_change(self, len_):
        self.__len = len_

    def valueOf(self, index):
        node = self._objectOf(index)
        return node.value()

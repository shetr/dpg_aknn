
#include <functional>
#include <gtest/gtest.h>
#include <aknn/pri_queue.h>

#include "test_data.h"

template<typename PriQueueT>
void TestPriQueue()
{
    PriQueueT queue;
    queue.Init(5, std::less<int>{});
    EXPECT_TRUE(queue.IsEmpty());
    EXPECT_TRUE(!queue.IsFull());
    EXPECT_EQ(0, queue.GetSize());
    
    // push {9, 2, 5, 10, 3, 1, 7, 8, 4, 6}

    queue.Push(9);
    EXPECT_EQ(9, queue.GetFirst());
    EXPECT_EQ(9, queue.GetLast());
    EXPECT_TRUE(!queue.IsEmpty());
    EXPECT_TRUE(!queue.IsFull());
    EXPECT_EQ(1, queue.GetSize());
    
    queue.Push(2);
    EXPECT_EQ(2, queue.GetFirst());
    EXPECT_EQ(9, queue.GetLast());
    EXPECT_TRUE(!queue.IsEmpty());
    EXPECT_TRUE(!queue.IsFull());
    EXPECT_EQ(2, queue.GetSize());
    
    queue.Push(5);
    EXPECT_EQ(2, queue.GetFirst());
    EXPECT_EQ(9, queue.GetLast());
    EXPECT_TRUE(!queue.IsEmpty());
    EXPECT_TRUE(!queue.IsFull());
    EXPECT_EQ(3, queue.GetSize());
    
    queue.Push(10);
    EXPECT_EQ(2, queue.GetFirst());
    EXPECT_EQ(10, queue.GetLast());
    EXPECT_TRUE(!queue.IsEmpty());
    EXPECT_TRUE(!queue.IsFull());
    EXPECT_EQ(4, queue.GetSize());
    
    queue.Push(3);
    EXPECT_EQ(2, queue.GetFirst());
    EXPECT_EQ(10, queue.GetLast());
    EXPECT_TRUE(!queue.IsEmpty());
    EXPECT_TRUE(queue.IsFull());
    EXPECT_EQ(5, queue.GetSize());
    
    queue.Push(1);
    EXPECT_EQ(1, queue.GetFirst());
    EXPECT_EQ(9, queue.GetLast());
    EXPECT_TRUE(!queue.IsEmpty());
    EXPECT_TRUE(queue.IsFull());
    EXPECT_EQ(5, queue.GetSize());
    
    queue.Push(7);
    EXPECT_EQ(1, queue.GetFirst());
    EXPECT_EQ(7, queue.GetLast());
    EXPECT_TRUE(!queue.IsEmpty());
    EXPECT_TRUE(queue.IsFull());
    EXPECT_EQ(5, queue.GetSize());
    
    queue.Push(8);
    EXPECT_EQ(1, queue.GetFirst());
    EXPECT_EQ(7, queue.GetLast());
    EXPECT_TRUE(!queue.IsEmpty());
    EXPECT_TRUE(queue.IsFull());
    EXPECT_EQ(5, queue.GetSize());
    
    queue.Push(4);
    EXPECT_EQ(1, queue.GetFirst());
    EXPECT_EQ(5, queue.GetLast());
    EXPECT_TRUE(!queue.IsEmpty());
    EXPECT_TRUE(queue.IsFull());
    EXPECT_EQ(5, queue.GetSize());
    
    queue.Push(6);
    EXPECT_EQ(1, queue.GetFirst());
    EXPECT_EQ(5, queue.GetLast());
    EXPECT_TRUE(!queue.IsEmpty());
    EXPECT_TRUE(queue.IsFull());
    EXPECT_EQ(5, queue.GetSize());

    // result
    std::vector<int> expected = {1, 2, 3, 4, 5};
    std::vector<int> result = queue.GetValues();
    std::sort(result.begin(), result.end());
    EXPECT_EQ(expected, result);
}

TEST(LinearPriQueueInt, basicInts) {
    TestPriQueue<LinearPriQueue<int>>();
}

TEST(StdPriQueueInt, basicInts) {
    TestPriQueue<StdPriQueue<int>>();
}

TEST(HeapPriQueueInt, basicInts) {
    TestPriQueue<HeapPriQueue<int>>();
}
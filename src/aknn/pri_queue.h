#ifndef AKNN_PRI_QUEUE_H
#define AKNN_PRI_QUEUE_H

#include <vector>
#include <functional>
#include <algorithm>

//! Shared interface for priority queues of fixed size. Used for k-NN implementation.
//! The implementations should behave like normal priority queue,
//! with the difference that it should remove elements with lowest priority if the capacity exceeds specified maximum.
template<typename T>
class FixedPriQueue
{
public:
    //! Functor used to mantain the queue priority. Should return true if left parameter is smaller than the right parameter.
    using IsLeftSmaller = std::function<bool(const T&, const T&)>;

    //! Clears previous state of the queue and initializes empty queue of size k
    virtual void Init(int k, IsLeftSmaller isLeftSmaller) = 0;
    //! Gets element with the highest priority.
    virtual const T& GetFirst() const = 0;
    //! Gets element with the lowest priority
    virtual const T& GetLast() const = 0;
    //! True if the queue is empty
    virtual bool IsEmpty() const = 0;
    //! True if the queue has reached the maximum size
    virtual bool IsFull() const = 0;
    //! Number of elements currently stored in the queue
    virtual int GetSize() const = 0;
    //! Return unsorted array of the values currently stored in the queue
    virtual std::vector<T> GetValues() const = 0;
    //! Push specified element into the queue. If queue is full, it removes element with the lowest priority.
    virtual void Push(const T& value) = 0;
};

template<typename T>
class LinearPriQueue : public FixedPriQueue<T>
{
private:
    std::vector<T> _values;
    int _k;
    IsLeftSmaller _isLeftSmaller;
    T* _first = nullptr;
    T* _last = nullptr;
public:
    //! Clears previous state of the queue and initializes empty queue of size k
    void Init(int k, IsLeftSmaller isLeftSmaller) override {
        _values.clear();
        _values.reserve(k);
        _k = k;
        _isLeftSmaller = isLeftSmaller;
    }
    //! Gets element with the highest priority
    const T& GetFirst() const override { return *_first; }
    //! Gets element with the lowest priority
    const T& GetLast() const override { return *_last; }
    //! True if the queue is empty
    bool IsEmpty() const override { return _values.empty(); }
    //! True if the queue has reached the maximum size
    bool IsFull() const override { return _values.size() == _k; }
    //! Number of elements currently stored in the queue
    int GetSize() const override { return _values.size(); }
    //! Return unsorted array of the values currently stored in the queue
    std::vector<T> GetValues() const override { return _values; }
    //! Push specified element into the queue. If queue is full, it removes element with the lowest priority.
    void Push(const T& value) override {
        if (IsEmpty()) {
            _values.push_back(value);
            _last = _values.data();
            _first = _values.data();
            return;
        }
        if (!IsFull()) {
            _values.push_back(value);
            if (_isLeftSmaller(*_last, value))
                _last = &_values.back();
            else if (_isLeftSmaller(value, *_first))
                _first = &_values.back();
            return;
        }
        // insert only if the value is smaller than largest element
        if (_isLeftSmaller(value, *_last)) {
            *_last = value;
            if (_isLeftSmaller(value, *_first))
                _first = _last;
            // find new largest value
            for (int i = 0; i < GetSize(); ++i)
                if (_isLeftSmaller(*_last, _values[i]))
                    _last = &_values[i];
        }
    }
};

template<typename T>
class HeapPriQueue : public FixedPriQueue<T>
{
private:
    std::vector<T> _heap;
    int _k;
    IsLeftSmaller _isLeftSmaller;
    T _first;
public:
    //! Clears previous state of the queue and initializes empty queue of size k
    void Init(int k, IsLeftSmaller isLeftSmaller) override {
        _heap.clear();
        _heap.reserve(k);
        _k = k;
        _isLeftSmaller = isLeftSmaller;
    }
    //! Gets element with the highest priority
    const T& GetFirst() const override { return _first; }
    //! Gets element with the lowest priority
    const T& GetLast() const override { return _heap.front(); }
    //! True if the queue is empty
    bool IsEmpty() const override { return _heap.empty(); }
    //! True if the queue has reached the maximum size
    bool IsFull() const override { return _heap.size() == _k; }
    //! Number of elements currently stored in the queue
    int GetSize() const override { return _heap.size(); }
    //! Return unsorted array of the values currently stored in the queue
    std::vector<T> GetValues() const override { return _heap; }
    //! Push specified element into the queue. If queue is full, it removes element with the lowest priority.
    void Push(const T& value) override {
        if (IsEmpty()) {
            _heap.push_back(value);
            _first = value;
        } else if (!IsFull()) {
            int i = GetSize();
            _heap.push_back(value);
            if (_isLeftSmaller(value, _first)) {
                _first = value;
            } else {
                // ensure the heap property that parent is larger
                while (i > 0) {
                    int p = GetParent(i);
                    if (_isLeftSmaller(_heap[i], _heap[p]))
                        break;
                    else
                        std::swap(_heap[i], _heap[p]);
                    i = p;
                }
            }
        } else if (_isLeftSmaller(value, _heap[0])) { // insert only if the value is smaller than largest element
            _heap[0] = value;
            // update first
            if (_isLeftSmaller(value, _first)) {
                _first = value;
            }
            // ensure the heap property that parent is larger than its childs
            int i = 0;
            for (int l = GetLeftChild(i); l < GetSize(); l = GetLeftChild(i)) {
                int r = l + 1;
                int larger = r < GetSize() ? (_isLeftSmaller(_heap[r], _heap[l]) ? l : r) : l;
                if (_isLeftSmaller(_heap[larger], _heap[i]))
                    break;
                else
                    std::swap(_heap[i], _heap[larger]);
                i = larger;
            }
        }
    }
private:
    int GetParent(int i) const { return (i - 1) >> 1; }
    int GetLeftChild(int i) const { return (i << 1) + 1; }
};

template<typename T>
class StdPriQueue : public FixedPriQueue<T>
{
private:
    std::vector<T> _heap;
    int _k;
    IsLeftSmaller _isLeftSmaller;
    T _first;
public:
    //! Clears previous state of the queue and initializes empty queue of size k
    void Init(int k, IsLeftSmaller isLeftSmaller) override {
        _heap.clear();
        _heap.reserve(k);
        _k = k;
        _isLeftSmaller = isLeftSmaller;
    }
    //! Gets element with the highest priority
    const T& GetFirst() const override { return _first; }
    //! Gets element with the lowest priority
    const T& GetLast() const override { return _heap.front(); }
    //! True if the queue is empty
    bool IsEmpty() const override { return _heap.empty(); }
    //! True if the queue has reached the maximum size
    bool IsFull() const override { return _heap.size() == _k; }
    //! Number of elements currently stored in the queue
    int GetSize() const override { return _heap.size(); }
    //! Return unsorted array of the values currently stored in the queue
    std::vector<T> GetValues() const override { return _heap; }
    //! Push specified element into the queue. If queue is full, it removes element with the lowest priority.
    void Push(const T& value) override {
        if (IsEmpty() || _isLeftSmaller(value, _first)) {
            _first = value;
        }
        _heap.push_back(value);
        std::push_heap(_heap.begin(), _heap.end(), _isLeftSmaller);
        if (_heap.size() > _k) {
            std::pop_heap(_heap.begin(), _heap.end(), _isLeftSmaller);
            _heap.pop_back();
        }
    }
};

#endif // AKNN_PRI_QUEUE_H
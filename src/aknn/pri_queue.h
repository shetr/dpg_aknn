#ifndef AKNN_PRI_QUEUE_H
#define AKNN_PRI_QUEUE_H

#include <vector>
#include <functional>
#include <algorithm>

template<typename T>
class FixedPriQueue
{
public:
    using IsLeftSmaller = std::function<bool(const T&, const T&)>;

    virtual void Init(int k, IsLeftSmaller isLeftSmaller) = 0;
    virtual const T& GetFirst() const = 0;
    virtual const T& GetLast() const = 0;
    virtual bool IsEmpty() const = 0;
    virtual bool IsFull() const = 0;
    virtual int GetSize() const = 0;
    virtual std::vector<T> GetValues() const = 0;
    virtual void Push(const T& value) = 0;
};

template<typename T>
class LinearPriQueue : public FixedPriQueue<T>
{
private:
    std::vector<T> _values;
    int _k;
    IsLeftSmaller _isLeftSmaller;
    T* _first;
    T* _last;
public:
    void Init(int k, IsLeftSmaller isLeftSmaller) override {
        _values.clear();
        _values.reserve(k);
        _k = k;
        _isLeftSmaller = isLeftSmaller;
    }
    const T& GetFirst() const override { return *_first; }
    const T& GetLast() const override { return *_last; }
    bool IsEmpty() const override { return _values.empty(); }
    bool IsFull() const override { return _values.size() == _k; }
    int GetSize() const override { return _values.size(); }
    std::vector<T> GetValues() const override { return _values; }
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
    T* _first;
public:
    void Init(int k, IsLeftSmaller isLeftSmaller) override {
        _heap.clear();
        _heap.reserve(k);
        _k = k;
        _isLeftSmaller = isLeftSmaller;
    }
    const T& GetFirst() const override { return *_first; }
    const T& GetLast() const override { return _heap.front(); }
    bool IsEmpty() const override { return _heap.empty(); }
    bool IsFull() const override { return _heap.size() == k; }
    int GetSize() const override { return _heap.size(); }
    std::vector<T> GetValues() const override { return _heap; }
    void Push(const T& value) override {
        if (IsEmpty()) {
            _heap.push_back(value);
            _first = _heap.data();
            return;
        }
        if (!IsFull()) {
            int i = GetSize();
            _heap.push_back(value);
            if (_isLeftSmaller(value, *_first)) {
                _first = &_heap[i];
                return;
            }
            // ensure the heap property that parent is larger
            while (i > 0) {
                int p = GetParent(i);
                if (_isLeftSmaller(_heap[i], _heap[p]))
                    break;
                else
                    std::swap(_heap[i], _heap[p]);
                i = p;
            }
            return;
        }
        // insert only if the value is smaller than largest element
        if (_isLeftSmaller(value, _heap[0])) {
            _heap[0] = value;
            int i = 0;
            // ensure the heap property that parent is larger than its childs
            for (int l = GetLeftChild(i); l < GetSize(); l = GetLeftChild(i)) {
                int r = l + 1;
                int larger = _isLeftSmaller(_heap[r], _heap[l]) ? l : r;
                if (_isLeftSmaller(_heap[larger], _heap[i]))
                    break;
                else
                    std::swap(_heap[i], _heap[larger]);
                i = larger;
            }
            // update first
            if (_isLeftSmaller(value, *_first)) {
                _first = &_heap[i];
            }
        }
    }
private:
    int GetParent(int i) const { return (i - 1) >> 1; }
    int GetLeftChild(int i) const { return (i << 1) + 1; }
};

#endif // AKNN_PRI_QUEUE_H
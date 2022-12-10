#ifndef AKNN_PRI_QUEUE_H
#define AKNN_PRI_QUEUE_H

#include <vector>
#include <functional>
#include <algorithm>

template<typename T>
class FixedPriQueue
{
public:
    using Compare = std::function<bool(const T&, const T&)>;

    virtual void Init(int k, Compare compare) = 0;
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
    Compare _compare;
public:
    void Init(int k, Compare compare) override {
        _values.clear();
        _k = k;
        _compare = compare;
        _values.reserve(k);
    }
    const T& GetFirst() const override { return _values.front(); }
    const T& GetLast() const override { return _values.back(); }
    bool IsEmpty() const override { return _values.empty(); }
    bool IsFull() const override { return _values.size() == k; }
    int GetSize() const override { return _values.size(); }
    std::vector<T> GetValues() const override { return _values; }
    void Push(const T& value) override {
        bool inserted = false;
        for (int i = 0; i < GetSize(); ++i) {
            if (_compare(value, _values[i])) {
                _values.insert(_values.begin() + i, value);
                inserted = true;
                break;
            }
        }
        if (!inserted && GetSize() < k) {
            _values.push_back(value);
        }
        if (GetSize() > _k) {
            _values.pop_back();
        }
    }
};

#endif // AKNN_PRI_QUEUE_H
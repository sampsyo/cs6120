+++
title = "Double-Checked Locking is Broken"
extra.author = "Hongbo Zhang"
extra.author_link = "https://www.cs.cornell.edu/~hongbo/"
extra.bio = """Hongbo Zhang is a first PhD student in computer science. He is interested in systems and computer architecture. He is also an okay archer shooting recurve bow."""
+++

Double-checked locking is a software design pattern for 
reducing the overhead of acquiring a lock.
The program checks locking criteria first, 
and acquires the lock only if the check indicates that locking is required.

Typically, double-checked locking is used for **lazy initialization**
in multi-threaded environments,
which is a commonly used tactic for delaying the object initialization 
until the first time it is accessed.
Since only the first access requires locking,
double-checked locking can be used to avoid locking overhead of subsequent accesses.

### Single-threading lazy initialization won't work in multi-threading

The lazy initialization should look like the following:

```Java
class Foo {
    private Helper helper = null;
    public Helper getHelper() {
        if (helper == null)
            helper = new Helper();
        return helper;
    }
}
```

This code works for a single thread, but if the code is run in multithreading,
    two or more threads could find that `helper` is `null` at the same time, 
    and create multiple copies of `Helper` object.
This can even cause a memory leak in some languages, such as C++.

### Always-synchronized solution is slow

To fix this issue, we can simply add a lock to this critical section as following,
   so that only one thread can enter this critical section at a time.

```Java
class Foo {
    private Helper helper = null;
    public Helper getHelper() {
        synchronized(this) {
            if (helper == null)
                helper = new Helper();
            return helper;
        }
    }
}
```

However, we only need this section of code to be synchronized for the first time,
    after the object is created, acquiring and releasing lock is unnecessary,
    which have a huge performance impact.

### Double-checked locking is broken 

What we want is like the following:

1. Check if the object is initialized without locking.
If it is, then return the object immediately.
2. Acquire the lock and check again if the object is initialized.
If another thread has grabbed the lock before, 
   the current thread can see the object is created, and return the object.
3. Otherwise, the current thread will create the object and return.

With the guideline above, we will get a code like following:

```Java
class Foo {
    private Helper helper = null;
    public Helper getHelper() {
        if (helper == null) {              // first check
            synchronized(this) {
                if (helper == null)         // second check
                    helper = new Helper();
            }
        }
        return helper;
    }
}
```

Intuitively, this is a simple fix, but this code is not guaranteed to work.
Some compilers, memory systems, or processors may reorder the instructions,
     so that the initialization of the object is done after the assignment of the pointer to the object.
     Another thread may see that the pointer is assigned and access the fields of the object before the fields of the object is initialized.

### Another fix is also broken

Some people came up with another fix for this issue with a memory barrier.
A memory barrier is a type of instruction that can make the compiler and processor
enforce the ordering, so that the instructions on one side of the memory barrier will
not be reordered to the other side of the barrier.

```Java
class Foo {
    private Helper helper = null;

    public Helper getHelper() {
        if (helper == null) {
            Helper h;
            synchronized(this) {
                h = helper;
                if (h == null)
                    synchronized(this) {
                        h = new Helper();
                    }                       
                helper = h;
            }
        }
        return helper;
    }
}

```

The intuitive idea is that the lock releasing would act as a memory barrier,
so that `helper=h` will not be executed until the initialization 
in the synchronized section is done.

Unfortunately, the lock releasing is a one-way memory barrier on many processors.
It only enforces that the instructions in the synchronized section 
must be executed before lock is released.
The instruction `helper=h` behind the memory barrier could still be moved into 
synchronized section and executed before the object initialization is done.

## Working Solutions


### Volatile in JDK5

Java introduced "volatile" in JDK5. 
A `volatile` qualifier makes the variable be stored and accessed in main memory.
Every read of a volatile is from main memory.
Every write of a volatile is to main memory.

The "volatile" also provides a "happen-before" guarantee:

* The reads/writes of other variables before a write to a volatile variable 
cannot be reordered after the write to the volatile variable.
* The reads/writes of other variables after a read from a volatile variable
cannot be reordered before the read from the volatile variable.

With this new feature, the double-checked locking issue is resolved by simply changing
the `helper` to volatile, since it guarantees that the initialization of the
object `new Helper()` happens before the assignment to the `helper`.

```Java
class Foo {
    private volatile Helper helper = null;
    public Helper getHelper() {
        if (helper == null) {
            synchronized(this) {    
                if (helper == null)
                    helper = new Helper();
            }
        }
        return helper;
    }
}
```

However, since all read and write operations of a volatile variable need to
access main memory, which can be very slow. 
An improvement can be done with a local variable, to reduce number of times 
accessing main memory.

```Java
class Foo {
    private volatile Helper helper = null;
    public Helper getHelper() {
        Helper h = helper;
        if (h == null) {
            synchronized(this) {    
                h = helper;
                if (h == null) {
                    h = new Helper();
                    helper = h;
                }
            }
        }
        return h;
    }
}

```

In cases that the `helper` in already initialized, 
   this optimization can reduce one volatile read by returning the local variable.

### Explicit Memory Barrier

The last fix with two synchronized sections does not work because releasing lock
is an implicit "one-way" memory barrier.
It is possible to make the double-checked locking actually work with an
explicit memory barrier.
For example, in C++11 we can safely implement double-checked locking with 
`std::atomic` and `std::atomic_thread_fence`.

```C++
class Foo {
    private:
        std::atomic <Foo*> helper;
    public:
        Foo* get_helper() {
            Foo* h = helper.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);        //memory barrier
            if (h == nullptr) {
                std::lock_guard<std::mutex> lock(m_init);
                h = helper.load(std::memory_order_relaxed);
                if (h == nullptr) {
                    h = new Helper;
                    std::atomic_thread_fence(std::memory_order_release);//memory barrier
                    helper.store(h, std::memory_order_relaxed);
                }
            }
            return tmp;
        }
};
```

* `std::atomic_thread_fence(std::memory_order_acquire)` guarantees that 
read/write after a memory barrier cannot be reordered before the memory barrier.
* `std::atomic_thread_fence(std::memory_order_release)` guarantees that 
read/write before a memory barrier cannot be reordered after the memory barrier.

Similar to the Java volatile case, 
the initialization must be done before it stores `h` to `helper` 
with the memory barrier.

### Static Singleton

If the `helper` is static, i.e., all the instances of class `Foo` share the 
same instance of `helper`, defining the `helper` in a static field of a separate
class will solve the problem.

```Java
class Foo {
    private static class HelperSingleton {
        public static final Helper helper = new Helper();
    }

    public Helper getHelper() {
        return HelperSingleton.helper;
    }
}
```

This is known as **initialization-on-demand holder idiom**, 
which is a safe and efficient concurrent lazy initialization 
for all Java versions.

* **Lazy Initialization**: as specified by the Java language,
    the static class `HelperSingleton` is not initialized until the first it is
    referenced.
* **Safe**: the class initialization is guaranteed to be synchronized by 
    Java language.
* **Efficient**: all subsequent calls of `getHelper` will return the initialized
    `helper` without any synchronization overhead.

### Thread Local

`ThreadLocal` is a variable that each thread will have its own copy of its value.
Each thread can only access and modify its own copy of a thread local variable
independently of other threads.

A thread local can be used to maintain the state of "whether the state has gone
through the synchronized initialization". 
If a thread has gone through the synchronized initialization once,
it can be confident that that object is already initialized.

Inside the synchronized initialization section, 
only the first thread will find the object is `null` and initialized object.
All threads will change their per-thread state, 
    so that they will not enter the synchronized section again.

```Java
class Foo {
    private static ThreadLocal perThreadState = new ThreadLocal();
    private Helper helper = null;
    public Helper getHelper() {
        if (perThreadState.get() == null) {
            synchronized {
                if (helper == null)
                    helper = new Helper();
                perThreadState.set(perThreadState);
            }
        }
        return helper;
    }
}
```

### 32-bit Primitives

If the initialized value is a 32-bit primitive value, such as `int` or `float`,
   the double-checked locking will work.

```Java
class Foo {
    private int magicNumber = 0;
    public int getMagicNumber() {
        if (magicNumber == 0) {
            synchronized(this) {
                if (magicNumber == 0)
                    magicNumber = TheMagicNumber();
            }
        }
        return helper;
    }
}
```

The assignment to 32-bit primitives can only happen once the data is available.
And the assignment is atomic. 
Other threads will only see a ready-to-use value or 0,
      there is no intermediate state of "initializing".

In fact, if the function `TheMagicNumber` always returns the same value without
side effects, the locking is not required.

```Java
class Foo {
    private int magicNumber = 0;
    public int getMagicNumber() {
        if (magicNumber == 0) 
            magicNumber = TheMagicNumber();
        return helper;
    }
}
```

## Conclusion

The article discussed the problem of double-checked locking for lazy 
initialization in multi-threaded environments.
It analyzed why some intuitive solutions do not work,
   and also analyzed some working solutions.

Writing multi-threaded program is hard. 
Writing correct and safe multi-threaded program is even harder.
When analyzing the correctness of multi-threaded programs,
it requires the considerations of multiple components, 
including compilers, systems, and processors.
On the other hand, when designing compilers, systems, or processors,
one also needs to take into consideration commonly used design patterns.

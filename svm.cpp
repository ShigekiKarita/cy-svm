#include "svm.h"

#include <cmath>
// #include <iostream>
// #include <opencv2/opencv.hpp>



using namespace std;


void SVM::learning(const vector< vector<double> >& train_set, const vector<double>& target, const size_t loop_limit = 1000)
{
    // 初期化
    a = decltype(a)(train_set.size());
    w = decltype(w)(train_set.size());
    err_cache = decltype(err_cache)(train_set.size());
    sv_index = decltype(sv_index)();
    m_train_set = train_set;
    m_y = target;
    train_set_size = train_set.size();
    data_size = train_set[0].size();
    dist = decltype(dist)(-1.0, 1.0);
    
    // 初期値設定
    fill(a.begin(), a.end(), 0.0);
    fill(err_cache.begin(), err_cache.end(), 0.0);
    b = 0.0;
    
    bool alldata = true;
    
    for (size_t loop = 0; loop < loop_limit; ++loop)
    {
        size_t changed = 0;
        
        for (size_t i = 0; i < train_set_size; ++i)
        {
            if (alldata || (a[i] > eps && a[i] < (C - eps)))
            {
                changed += examinUpdate(i);
            }
        }
        
        if (alldata)
        {
            alldata = false;
            
            if (changed == 0) break;
        }
        else if (changed == 0)
        {
            alldata = true;
        }
        
        // printf("loop %zu : changed %zu\n", loop, changed);
    }
    
    sv_index.reserve(train_set_size);
    for (size_t i = 0; i < train_set_size; i++)
    {
        if (a[i] != 0.0)
        {
            sv_index.push_back(static_cast<int>(i));
        }
    }
    sv_index.shrink_to_fit();
    
    // 重みw 計算
    for (auto it = sv_index.begin(); it != sv_index.end(); it++)
    {
        w[*it] = m_y[*it] * a[*it];
    }
}

size_t SVM::examinUpdate(const size_t i)
{
    double yFi;
    
    if (a[i] > eps && a[i] < (C - eps))
    {
        Ei = err_cache[i];  // f(x)-y
    }
    else
    {
        Ei = f(i) - m_y[i];
    }
    yFi = Ei * m_y[i];      // yf(x)-1
    
    // KKT条件のチェック
    if ((a[i] < (C - eps) && yFi < -tolerance) || (a[i] > eps && yFi > tolerance))
    {
        return update(i);
    }
    
    return 0;
}

size_t SVM::update(const size_t i)
{
    // a[j] の決定
    double max_Ej = 0.0;
    bool is_found = false;
    size_t max_j = 0;
    
    // 1
    int offset = static_cast<int>(dist(rand_engine) * (train_set_size - 1));
    
    for (size_t j = 0; j < train_set_size; j++)
    {
        size_t pos = (j + offset) % train_set_size;
        
        if (a[pos] > eps && a[pos] < (C - eps))
        {
            double Ej = err_cache[pos];
            
            if (fabs(Ej - Ei) > max_Ej)
            {
                max_Ej = fabs(Ej - Ei);
                max_j = pos;
                is_found = true;
            }
        }
    }
    
    if (is_found && stepSMO(i, max_j) == 1)
    {
        return 1;
    }
    
    // 2
    offset = dist(rand_engine) * (train_set_size - 1);
    
    for (int j = 0; j < train_set_size; j++)
    {
        int pos = (j + offset) % train_set_size;
        
        if (a[pos] > eps && a[pos] < (C - eps) && stepSMO(i, pos) == 1)
        {
            return 1;
        }
    }
    
    // 3
    offset = dist(rand_engine) * (train_set_size - 1);
    for (int j = 0; j < train_set_size; j++)
    {
        int pos = (j + offset) % train_set_size;
        
        if (!(a[pos] > eps && a[pos] < (C - eps)) && stepSMO(i, pos) == 1)
        {
            return 1;
        }
    }
    
    return 0;
}

size_t SVM::stepSMO(const size_t i, const size_t j)
{
    if (i == j) return 0;
    
    const double ai_old = a[i];
    const double aj_old = a[j];
    double ai_new;
    double aj_new;
    double U, V;

    if (m_y[i] != m_y[j])
    {
        U = max(0.0, ai_old - aj_old);
        V = min(C, C + ai_old - aj_old);
    }
    else
    {
        U = max(0.0, ai_old + aj_old - C);
        V = min(C, ai_old + aj_old);
    }
    
    if (U == V) return 0;
    
    const double kii = kernel(m_train_set[i], m_train_set[i]);
    const double kjj = kernel(m_train_set[j], m_train_set[j]);
    const double kij = kernel(m_train_set[i], m_train_set[j]);
    const double k = kii + kjj - 2.0*kij;

    if (a[j] > eps && a[j] < (C - eps))
    {
        Ej = err_cache[j];
    }
    else
    {
        Ej = f(j) - m_y[j];
    }
    
    bool bClip = false;
    
    if (k <= 0.0)
    {
        // ai = U のときの目的関数の値
        ai_new = U;
        aj_new = aj_old + m_y[i] * m_y[j] * (ai_old - ai_new);
        a[i] = ai_new; // 仮置き
        a[j] = aj_new;
        double v1 = f(j) + b - m_y[j] * aj_old * kjj - m_y[i] * ai_old * kij;
        double v2 = f(i) + b - m_y[j] * aj_old * kij - m_y[i] * ai_old * kii;
        double Lobj = aj_new + ai_new - kjj * aj_new * aj_new / 2.0 - kii * ai_new * ai_new / 2.0
            - m_y[j] * m_y[i] * kij * aj_new * ai_new
            - m_y[j] * aj_new * v1 - m_y[i] * ai_new * v2;
        // ai = V のときの目的関数の値
        ai_new = V;
        aj_new = aj_old + m_y[i] * m_y[j] * (ai_old - ai_new);
        a[i] = ai_new; // 仮置き
        a[j] = aj_new;
        v1 = f(j) + b - m_y[j] * aj_old * kjj - m_y[i] * ai_old * kij;
        v2 = f(i) + b - m_y[j] * aj_old * kij - m_y[i] * ai_old * kii;
        double Hobj = aj_new + ai_new - kjj * aj_new * aj_new / 2.0 - kii * ai_new * ai_new / 2.0
            - m_y[j] * m_y[i] * kij * aj_new * ai_new
            - m_y[j] * aj_new * v1 - m_y[i] * ai_new * v2;
        
        if (Lobj > Hobj + eps)
        {
            bClip = true;
            ai_new = U;
        }
        else if (Lobj < Hobj - eps)
        {
            bClip = true;
            ai_new = V;
        }
        else
        {
            bClip = true;
            ai_new = ai_old;
        }
        a[i] = ai_old; // 元に戻す
        a[j] = aj_old;
    }
    else
    {
        ai_new = ai_old + (m_y[i] * (Ej - Ei) / k);
        if (ai_new > V)
        {
            bClip = true;
            ai_new = V;
        }
        else if (ai_new < U)
        {
            bClip = true;
            ai_new = U;
        }
    }
    if (fabs(ai_new - ai_old) < eps * (ai_new + ai_old + eps))
    {
        return 0;
    }
    
    // a[j]更新
    aj_new = aj_old + m_y[i] * m_y[j] * (ai_old - ai_new);
    // b更新
    double old_b = b;
    if (a[i] > eps && a[i] < (C - eps))
    {
        b += Ei + (ai_new - ai_old) * m_y[i] * kii +
            (aj_new - aj_old) * m_y[j] * kij;
    }
    else if (a[j] > eps && a[j] < (C - eps))
    {
        b += Ej + (ai_new - ai_old) * m_y[i] * kij +
            (aj_new - aj_old) * m_y[j] * kjj;
    }
    else
    {
        b += (Ei + (ai_new - ai_old) * m_y[i] * kii +
              (aj_new - aj_old) * m_y[j] * kij +
              Ej + (ai_new - ai_old) * m_y[i] * kij +
              (aj_new - aj_old) * m_y[j] * kjj) / 2.0;
    }
    // err更新
    for (int m = 0; m < train_set_size; m++)
    {
        if (m == i || m == j) continue;

        else if (a[m] > eps && a[m] < (C - eps))
        {
            err_cache[m] = err_cache[m] + m_y[j] * (aj_new - aj_old) * kernel(m_train_set[j], m_train_set[m])
                + m_y[i] * (ai_new - ai_old) * kernel(m_train_set[i], m_train_set[m])
                + old_b - b;
        }
    }
    
    a[i] = ai_new;
    a[j] = aj_new;
    
    if (bClip && ai_new > eps && ai_new < C - eps)
    {
        err_cache[i] = f(i) - m_y[i];
    }
    else
    {
        err_cache[i] = 0.0;
    }
    err_cache[j] = f(j) - m_y[j];
    
    return 1;
}

double SVM::discriminate(const vector<double>& test)
{
    double ret = 0.0;

    for (auto it = sv_index.begin(); it != sv_index.end(); ++it)
    {
        ret += w[*it] * kernel(m_train_set[*it], test);
    }
    
    return ret - b;
}

double SVM::f(const size_t i)
{
    double ret = 0.0;
    
    for (int j = 0; j < train_set_size; j++)
    {
        if (a[j] == 0.0) continue;
        
        ret += a[j] * m_y[j] * kernel(m_train_set[j], m_train_set[i]);
    }
    
    return ret - b;
}

double SVM::kernel(const vector<double>& p, const vector<double>& q)
{
    double r = 0.0;
    if (this->is_linear)
    {
        for (size_t i = 0; i < data_size; ++i)
        {
            r += p[i] * q[i];
        }
    }
    else
    {
        for (size_t i = 0; i < data_size; i++)
        {
            r += (p[i] - q[i]) * (p[i] - q[i]);
        }
        r = exp(-r / 2);
    }
    return r;
}

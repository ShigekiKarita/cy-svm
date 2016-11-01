#include <vector>
#include <random>

using std::vector;


class SVM
{
public:
    void fit(double* train_set, double* target_set, size_t n, size_t d, size_t loop_limit) {
        // FIXME; set m_train_set, m_y instead of this
        auto&& xs = vector<vector<double>>(n);
        for (size_t i; i < n; ++i) {
            auto&& x = train_set + i * d;
            xs[i] = vector<double>(x, x + d);
        }

        auto&& ys = vector<double>(target_set, target_set + n);
        this->learning(xs, ys, loop_limit);
    }

    double predict(double* input) {
        return this->discriminate(vector<double>(input, input + this->data_size));
    }


private:
    void learning(const vector< vector<double> >& train_set,    // 教師データ群
                  const vector<double>& target,             // 正解出力 （-1 or 1)
                  const size_t loop_limit);
    double discriminate(const vector<double>& test_dataset);


    size_t examinUpdate(const size_t i);        // a[i]の更新評価(KKT条件のチェック)
    size_t update(const size_t i);              // a[i]との更新ペアa[j]を探す
    size_t stepSMO(const size_t i, const size_t j); // a[i]，a[j]を更新する
    double f(const size_t i);
    double kernel(const vector<double>& p, const vector<double>& q);
    
    vector<double> a;               // ラグランジュ乗数
    vector<double> w;               // 重み  a x target
    vector<int> sv_index;           // サポートベクトル(0でないa[])のインデックス
    vector<double> err_cache;       // エラー値のキャッシュ
    vector< vector<double> > m_train_set;   // 教師データ
    vector<double> m_y;         // 正解データ
    size_t train_set_size;              // 教師データの数量
    size_t data_size;                   // 教師データの要素数
    double b;                       // 閾値
    double C;                       //
    double eps;                     // ラグランジュ乗数評価時の余裕値
    double tolerance;               // KKT条件評価時の余裕値
    double Ei, Ej;                  // エラー値
    std::mt19937_64 rand_engine;
    std::uniform_real_distribution<double> dist;
};
#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
// #define endl endl << flush
#define endl '\n'
using namespace std;
#ifdef __linux__
#define gc getchar_unlocked
#define pc putchar_unlocked
#else
#define gc _getchar_nolock
#define pc _putchar_nolock
#endif
inline bool blank(const char x) { return !(x ^ 32) || !(x ^ 10) || !(x ^ 13) || !(x ^ 9); }
template <typename Tp>
inline void read(Tp &x)
{
    x = 0;
    bool z = true;
    char a = gc();
    for (; !isdigit(a); a = gc())
        if (a == '-')
            z = false;
    for (; isdigit(a); a = gc())
        x = (x << 1) + (x << 3) + (a ^ 48);
    x = (z ? x : ~x + 1);
}
inline void read(double &x)
{
    x = 0.0;
    bool z = true;
    double y = 0.1;
    char a = gc();
    for (; !isdigit(a); a = gc())
        if (a == '-')
            z = false;
    for (; isdigit(a); a = gc())
        x = x * 10 + (a ^ 48);
    if (a != '.')
        return x = z ? x : -x, void();
    for (a = gc(); isdigit(a); a = gc(), y /= 10)
        x += y * (a ^ 48);
    x = (z ? x : -x);
}
template <typename Tp>
inline void read(vector<Tp> &x)
{
    for (int i = 1; i < x.size(); i++)
        read(x[i]);
}
inline void read(char &x)
{
    for (x = gc(); blank(x) && (x ^ -1); x = gc())
        ;
}
inline void read(char *x)
{
    char a = gc();
    for (; blank(a) && (a ^ -1); a = gc())
        ;
    for (; !blank(a) && (a ^ -1); a = gc())
        *x++ = a;
    *x = 0;
}
inline void read(string &x)
{
    x = "";
    char a = gc();
    for (; blank(a) && (a ^ -1); a = gc())
        ;
    for (; !blank(a) && (a ^ -1); a = gc())
        x += a;
}
template <typename T, typename... Tp>
inline void read(T &x, Tp &...y) { read(x), read(y...); }
template <typename Tp>
inline void write(Tp x)
{
    if (!x)
        return pc(48), void();
    if (x < 0)
        pc('-'), x = ~x + 1;
    int len = 0;
    char tmp[64];
    for (; x; x /= 10)
        tmp[++len] = x % 10 + 48;
    while (len)
        pc(tmp[len--]);
}
inline void write(const double x)
{
    int a = 6;
    double b = x, c = b;
    if (b < 0)
        pc('-'), b = -b, c = -c;
    double y = 5 * powl(10, -a - 1);
    b += y, c += y;
    int len = 0;
    char tmp[64];
    if (b < 1)
        pc(48);
    else
        for (; b >= 1; b /= 10)
            tmp[++len] = floor(b) - floor(b / 10) * 10 + 48;
    while (len)
        pc(tmp[len--]);
    pc('.');
    for (c *= 10; a; a--, c *= 10)
        pc(floor(c) - floor(c / 10) * 10 + 48);
}
inline void write(const pair<int, double> x)
{
    int a = x.first;
    if (a < 7)
    {
        double b = x.second, c = b;
        if (b < 0)
            pc('-'), b = -b, c = -c;
        double y = 5 * powl(10, -a - 1);
        b += y, c += y;
        int len = 0;
        char tmp[64];
        if (b < 1)
            pc(48);
        else
            for (; b >= 1; b /= 10)
                tmp[++len] = floor(b) - floor(b / 10) * 10 + 48;
        while (len)
            pc(tmp[len--]);
        a && (pc('.'));
        for (c *= 10; a; a--, c *= 10)
            pc(floor(c) - floor(c / 10) * 10 + 48);
    }
    else
        cout << fixed << setprecision(a) << x.second;
}
inline void write(const char x) { pc(x); }
inline void write(const bool x) { pc(x ? 49 : 48); }
inline void write(char *x) { fputs(x, stdout); }
inline void write(const char *x) { fputs(x, stdout); }
inline void write(const string &x) { fputs(x.c_str(), stdout); }
template <typename Tp>
inline void write(const vector<Tp> &x)
{
    for (int i = 1; i < x.size(); i++)
        write(x[i]), i != x.size() - 1 && pc(' ');
    pc('\n');
}
template <typename T, typename... Tp>
inline void write(T x, Tp... y) { write(x), write(y...); }
template <typename Tp>
inline void wl(Tp x) { write(x), pc('\n'); }
inline void wl(const double x) { write(x), pc('\n'); }
inline void wl(const pair<int, double> x) { write(x), pc('\n'); }
inline void wl() { pc('\n'); }
template <typename T, typename... Tp>
inline void wl(T x, Tp... y) { wl(x), wl(y...); }
template <typename Tp>
inline void wr(Tp x) { write(x), pc(' '); }
inline void wr(const double x) { write(x), pc(' '); }
inline void wr(const pair<int, double> x) { write(x), pc(' '); }
inline void wr() { pc(' '); }
template <typename T, typename... Tp>
inline void wr(T x, Tp... y) { wr(x), wr(y...); }
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 1e6 + 6;
class FenwickTree
{
public:
    int n;
    vector<int> fen;
    FenwickTree(int n) : n(n), fen(n + 1) {}
    void upd(int x)
    {
        while (x <= n)
        {
            fen[x]++;
            x += x & -x;
        }
    }
    int que(int x)
    {
        int ans = 0;
        while (x)
        {
            ans += fen[x];
            x -= x & -x;
        }
        return ans;
    }
    int f(vector<int> x)
    {
        ff(i, 0, n) fen[i] = 0;
        int ans = 0;
        ffg(i, x.size() - 1, 0)
        {
            ans += que(x[i]);
            upd(x[i]);
        }

        return ans;
    }
};
void Murasame()
{
    int n, a, b, c;
    read(n, a, b, c);
    int u = (1LL << 30) - 1;
    int F1 = c & u, F2 = b & u, F3 = a & u, G, H, tp, L, R, D;
    int cnt = -1, res = 0, type = 0, cc = 0;
    if (n == 1)
    {
        wl(0);
        return;
    }
    while (cc < n - 1)
    {
        // wr(F1);
        G = F3 ^ (((1LL << 16) * F3) & u);
        H = G ^ (G / (1LL << 5));
        tp = H ^ ((2 * H) & u) ^ F2 ^ F1;
        F3 = F2;
        F2 = F1;
        F1 = tp;
        cnt++;
        if (cnt >= 0 && cnt <= n - 1)
        {
            if (F1 % (n - cnt) == 0)
                res++;
        }
        if (cnt == n)
        {
            type = (n - res) & 1;
            write(type);
        }
        if (cnt >= n + 2 && (cnt - n - 2) % 3 == 0)
        {
            // l[i] = min(f[n + 3 * i - 3] % n, f[n + 3 * i - 2] % n);
            // r[i] = max(f[n + 3 * i - 3] % n, f[n + 3 * i - 2] % n);
            // d[i] = ((f[n + 3 * i - 1] % n) + 1) % (r[i] - l[i] + 1);
            L = min(F3 % n, F2 % n);
            R = max(F3 % n, F2 % n);
            D = (F1 % n) + 1;
            // cerr << L << " " << R << " " << D << endl;
            cc++;
            if (((R - L) * D) & 1)
                type ^= 1;
            write(type);
        }
    }
    wl();
    // vi f(4 * n + 1), g(4 * n + 1), h(4 * n + 1), l(n), r(n), d(n);

    // g[0] = f3 ^ (((1LL << 16) * f3) & u);
    // g[1] = f2 ^ (((1LL << 16) * f2) & u);
    // g[2] = f1 ^ (((1LL << 16) * f1) & u);
    // h[0] = g[0] ^ (g[0] / (1LL << 5));
    // h[1] = g[1] ^ (g[1] / (1LL << 5));
    // h[2] = g[2] ^ (g[2] / (1LL << 5));
    // f[0] = h[0] ^ ((2 * h[0]) & u) ^ f2 ^ f1;
    // f[1] = h[1] ^ ((2 * h[1]) & u) ^ f1 ^ f[0];
    // f[2] = h[2] ^ ((2 * h[2]) & u) ^ f[0] ^ f[1];
    // ff(i, 3, 4 * n)
    // {
    //     g[i] = f[i - 3] ^ (((1LL << 16) * f[i - 3]) & u);
    //     h[i] = g[i] ^ (g[i] / (1LL << 5));
    //     f[i] = h[i] ^ ((2 * h[i]) & u) ^ f[i - 2] ^ f[i - 1];
    // }
    // ff(i, 1, n - 1)
    // {
    //     l[i] = min(f[n + 3 * i - 3] % n, f[n + 3 * i - 2] % n);
    //     r[i] = max(f[n + 3 * i - 3] % n, f[n + 3 * i - 2] % n);
    //     d[i] = ((f[n + 3 * i - 1] % n) + 1) % (r[i] - l[i] + 1);
    //     wr(l[i], r[i], d[i]);
    //     wl();
    // }
    // ff(i, 0, n - 1)
    // {
    //     swap(temp[i], temp[i + (f[i]) % (n - i)]);
    // }
    // ff(i, 0, n - 1)
    // {
    //     wr(temp[i]);
    // }
    // pc('\n');
}
signed main()
{
    //	ios::sync_with_stdio(false);
    //	cin.tie(0);
    //	cout.tie(0);
    int _T = 1;
    //
    read(_T);
    while (_T--)
    {
        Murasame();
    }
    return 0;
}
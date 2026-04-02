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
mt19937 rng((unsigned)chrono::steady_clock::now().time_since_epoch().count());
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

int dx[4] = {1, -1, 0, 0};
int dy[4] = {0, 0, 1, -1};

pii get(int n, vvi &a)
{
    vvi vis(n + 1, vi(n + 1, 0));
    int cnt0 = 0, cnt1 = 0;

    auto dfs = [&](auto &&self, int x, int y) -> void
    {
        vis[x][y] = 1;
        ff(d, 0, 3)
        {
            int tx = x + dx[d], ty = y + dy[d];
            if (tx < 1 || tx > n || ty < 1 || ty > n)
                continue;
            if (!vis[tx][ty] && a[tx][ty] == a[x][y])
            {
                self(self, tx, ty);
            }
        }
    };

    ff(i, 1, n)
    {
        ff(j, 1, n)
        {
            if (!vis[i][j])
            {
                if (a[i][j] == 0)
                    cnt0++;
                else
                    cnt1++;
                dfs(dfs, i, j);
            }
        }
    }

    return {cnt0, cnt1};
}

vvi init(int n)
{
    vvi res(n + 1, vi(n + 1, 0));
    ff(i, 1, n)
    {
        ff(j, 1, n)
        {
            if (n - i >= j)
                res[i][j] = 1;
        }
    }
    return res;
}
void sr(vvi &a, int m, int r1, int r2) //  换行
{
    if (r1 == r2)
        return;
    ff(j, 1, m)
    {
        swap(a[r1][j], a[r2][j]);
    }
}
void sc(vvi &a, int n, int c1, int c2) //  换列
{
    if (c1 == c2)
        return;
    ff(i, 1, n)
    {
        swap(a[i][c1], a[i][c2]);
    }
}
void debug_out(vvi &a, int n)
{
    cerr << get(n, a).first << ' ' << get(n, a).second << endl;
}
void print(vvi &a, int n)
{
    ff(i, 1, n)
    {
        ff(j, 1, n)
        {
            write(a[i][j]);
        }
        wl();
    }
}
void Murasame()
{
    int n;
    cin >> n;
    vvi a(n + 1, vi(n + 1, 0));
    for (int i = 1; i <= n; ++i)
    {
        int t = i % 2 == 0;
        for (int j = i; j <= n; ++j)
            a[i][j] = t;
        for (int j = i; j <= n; ++j)
            a[j][i] = t;
    }
    print(a, n);
}
signed main()
{
    //	ios::sync_with_stdio(false);
    //	cin.tie(0);
    //	cout.tie(0);
    int _T = 1;
    // read(_T);
    while (_T--)
    {
        Murasame();
    }
    return 0;
}
/*
1010
1011//4
1000
0000


01000
11010
01010//5
11011
00000


015243
011111 5
001111 4
001010 2
001011 3
001000 1
000000 0

*/
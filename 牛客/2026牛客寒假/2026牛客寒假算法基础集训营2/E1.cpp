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
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
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

int get(int n, vvi &a, vi &r, vi &c, int goal)
{
    int dx[4] = {1, -1, 0, 0};
    int dy[4] = {0, 0, 1, -1};

    bool vis[105][105];
    ff(i, 1, n) ff(j, 1, n) vis[i][j] = 0;

    int cnt = 0;

    ff(i, 1, n)
    {
        ff(j, 1, n)
        {
            if (vis[i][j])
                continue;
            cnt++;
            if (cnt > goal + 2) // jianzhi
                return cnt;
            int val = a[r[i]][c[j]];
            stack<pii> st;
            st.push({i, j});
            vis[i][j] = 1;

            while (!st.empty())
            {
                auto [x, y] = st.top();
                st.pop();
                for (int d = 0; d < 4; d++)
                {
                    int tx = x + dx[d], ty = y + dy[d];
                    if (tx < 1 || tx > n || ty < 1 || ty > n)
                        continue;
                    if (vis[tx][ty])
                        continue;
                    if (a[r[tx]][c[ty]] == val)
                    {
                        vis[tx][ty] = 1;
                        st.push({tx, ty});
                    }
                }
            }
        }
    }
    return cnt;
}
vi r(105), c(105);
bool solve(vvi &a, int n, int goal)
{

    ff(i, 0, n + 1)
    {
        r[i] = c[i] = i; // yingshe
    }

    int cur = get(n, a, r, c, goal);

    for (int it = 1; it <= 3e5; it++)
    {
        bool ok = rng() % 2;

        if (ok)
        {
            int x = rng() % n + 1;
            int y = rng() % n + 1;
            if (x == y)
                continue;
            swap(r[x], r[y]);
            int now = get(n, a, r, c, goal);
            if (now == goal)
                return true;
            if (abs(now - goal) <= abs(cur - goal)) // zhao li da a jing de
                cur = now;
            else
                swap(r[x], r[y]); // buxin jiuhuanhuiqu
        }
        else
        {
            int x = rng() % n + 1;
            int y = rng() % n + 1;
            if (x == y)
                continue;

            swap(c[x], c[y]);
            int now = get(n, a, r, c, goal);

            if (now == goal)
                return true;

            if (abs(now - goal) <= abs(cur - goal))
                cur = now;
            else
                swap(c[x], c[y]);
        }
    }
    return false;
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
void print(vvi &a, int n)
{
    ff(i, 1, n)
    {
        ff(j, 1, n)
        {
            write(a[r[i]][c[j]]);
        }
        wl();
    }
}
void Murasame()
{
    int n;
    read(n);
    if(n == 1)
    {
        write(0);
        return;
    }
    auto a = init(n);
    while (!solve(a, n, n))
    {
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
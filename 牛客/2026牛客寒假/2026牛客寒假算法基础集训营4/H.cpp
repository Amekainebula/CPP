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
const int N = 505;
int n, m, q;
int a[N][N];
int af[N][N];
struct Node
{
    int mx;
    int id;
} tr[N * N * 4];

int getid(int x, int y)
{
    return (x - 1) * m + y;
}

pair<int, int> gec(int id)
{
    return {(id - 1) / m + 1, (id - 1) % m + 1};
}

void push_up(int rt)
{
    if (tr[rt << 1].mx >= tr[rt << 1 | 1].mx)
    {
        tr[rt].mx = tr[rt << 1].mx;
        tr[rt].id = tr[rt << 1].id;
    }
    else
    {
        tr[rt].mx = tr[rt << 1 | 1].mx;
        tr[rt].id = tr[rt << 1 | 1].id;
    }
}

void build(int rt, int l, int r)
{
    if (l == r)
    {
        pair<int, int> coord = gec(l);
        tr[rt].mx = af[coord.first][coord.second];
        tr[rt].id = l;
        return;
    }
    int mid = (l + r) >> 1;
    build(rt << 1, l, mid);
    build(rt << 1 | 1, mid + 1, r);
    push_up(rt);
}

void update(int rt, int l, int r, int idx, int val)
{
    if (l == r)
    {
        tr[rt].mx += val;
        return;
    }
    int mid = (l + r) >> 1;
    if (idx <= mid)
        update(rt << 1, l, mid, idx, val);
    else
        update(rt << 1 | 1, mid + 1, r, idx, val);
    push_up(rt);
}

void Murasame()
{
    read(n, m, q);
    ff(i, 1, n) ff(j, 1, m)
    {
        read(a[i][j]);
    }

    ff(i, 1, n) ff(j, 1, m)
    {
        if (a[i][j] == 0)
            continue;
        ff(dx, -2, 2) ff(dy, -2, 2)
        {
            if (abs(dx) + abs(dy) <= 2)
            {
                int nx = i + dx, ny = j + dy;
                if (nx >= 1 && nx <= n && ny >= 1 && ny <= m)
                {
                    af[nx][ny] += a[i][j];
                }
            }
        }
    }
    build(1, 1, n * m);
    while (q--)
    {
        int x, y, z;
        read(x, y, z);
        int px = x + y, py = x - y + m;
        ff(dx, -2, 2) ff(dy, -2, 2)
        {
            int tx = px + dx, ty = py + dy;

            if ((tx + ty - m) % 2 == 0)
            {
                int nx = (tx + ty - m) / 2, ny = (tx - (ty - m)) / 2;
                if (nx >= 1 && nx <= n && ny >= 1 && ny <= m)
                {
                    update(1, 1, n * m, getid(nx, ny), z);
                }
            }
        }

        auto res = gec(tr[1].id);
        wr(res.first, res.second);
        pc('\n');
    }
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
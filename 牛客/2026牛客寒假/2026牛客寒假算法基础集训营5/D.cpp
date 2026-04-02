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
#define fi first
#define se second

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

struct cmp
{
    bool operator()(const pii &a, const pii &b) const
    {
        if (a.fi == b.fi)
            return a.se < b.se;
        return a.fi > b.fi;
    }
};
void Murasame()
{
    int ans = 0;
    int n;
    read(n);

    // priority_queue<pii, vector<pii>, cmp> q;
    map<int, int> mp;
    ff(i, 1, n)
    {
        int c, w;
        read(c, w);
        mp[w] += c;
    }
    while (!mp.empty())
    {
        auto it1 = mp.begin();
        // int w1 = it1->first;
        // int c1 = it1->second;
        auto [w1, c1] = *it1;
        if (mp.size() == 1 && c1 == 1)
            break;
        if (c1 >= 2)
        {
            int cnt = c1 / 2, val = w1 * 2;
            ans = (ans + ((cnt % MOD) * (val % MOD)) % MOD) % MOD;
            mp[val] += cnt;
            c1 %= 2;
            it1->second = c1;
            if (c1 == 0)
            {
                mp.erase(it1);
            }
        }
        if (!mp.empty() && mp.begin()->first == w1 && mp.begin()->second == 1)
        {
            if (mp.size() > 1)
            {
                auto it2 = next(mp.begin());
                auto [w2, c2] = *it2;
                int nw = w1 + w2;
                ans = (ans + (nw % MOD)) % MOD;
                mp[nw]++;
                mp.erase(w1);
                c2--;
                it2->second = c2;
                if (c2 == 0)
                {
                    mp.erase(it2);
                }
            }
            else
            {
                break;
            }
        }
    }
    wl(ans);
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
 int las = -1;
    while (q.size() > 1)
    {
        auto [w1, c1] = q.top();
        q.pop();
        if (mp[w1] == 0)
            continue;
        if (las != -1)
        {
            c1--;
            las = -1;
            int val = w1 + las;
            ans = (ans + val) % MOD;
            mp[val]++;
            q.push({val, mp[val]});
        }
        if (c1 & 1)
        {
            c1--;
            las = w1;
        }
        if (c1 == 0)
            continue;
        mp[w1] = 0;
        ans = (ans + (w1 * 2 * (c1 / 2)) % MOD) % MOD;
        q.push({w1 + w1, c1 / 2});
        mp[w1 + w1] += c1 / 2;
    }
    if(las!=-1)
    {

    }
    wl(ans);
*/
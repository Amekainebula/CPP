#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 1e6 + 6;
struct trie
{
    int nex[100005][26], cnt;
    bool exist[100005]; // 该结点结尾的字符串是否存在

    void insert(string s)
    { // 插入字符串
        int p = 0;
        for (int i = 0; i < s.length(); i++)
        {
            int c = s[i] - 'a';
            if (!nex[p][c])
                nex[p][c] = ++cnt; // 如果没有，就添加结点
            p = nex[p][c];
        }
        exist[p] = true;
    }

    bool find(string s)
    { // 查找字符串
        int p = 0;
        for (int i = 0; i < s.length(); i++)
        {
            int c = s[i] - 'a';
            if (!nex[p][c])
                return 0;
            p = nex[p][c];
        }
        return exist[p];
    }
};
void Murasame()
{
    int n, t, a;
    cin >> n >> t;
    // trie tr;
    // for (int i = 1; i <= n; i++)
    // {
    //     string s;
    //     cin >> s;
    //     tr.insert(s);
    // }
    string s, q;
    cin >> s;
    s = '1' + s;
    int cnt = 0;
    int ans = 0;
    while (t--)
    {
        ans = 0;
        cin >> q >> a;
        q = '1' + q;
        for (int i = 1; i < q.size(); i++)
        {
            if (s[a + i - 1] == q[i])
                cnt++;
            else
            {
                ans += cnt * (cnt + 1) / 2;
                cnt = 0;
            }
        }
        ans += cnt * (cnt + 1) / 2;
        cnt = 0;
        cout << ans << endl;
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}